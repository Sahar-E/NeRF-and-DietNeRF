import json
import os
import re
from pathlib import Path
from typing import Union, Dict

import imageio
import numpy as np
import yaml

from tensorflow.keras import Model
import tensorflow as tf
from src.ConfigurationKeys import GENERAL_SAVE_LOCATION, EXISTING_SAVE_DIR_NAME

# Formats:
from src.UtilsCV import recenter_poses, spherify_poses

GCP_PREFIX = "gs://"

POSES_BOUNDS_NPY = 'poses_bounds.npy'
SAVE_DIR_NAME_FORMAT = '{}_save_dir_{}'
SAVE_DIR_NAME_FORMAT_REGEX = r'^{}_save_dir_\d+$'

# File names:
CAM_DATA_JSON_FILE_NAME = 'cam_data.json'

# Metadata configuration keys:
METADATA_FOCAL_LENGTH_KEY = 'focal_length'
METADATA_FIELD_OF_VIEW = 'field_of_view'
METADATA_FRAME_METADATA = 'frames'
METADATA_FILENAME = 'filename'
METADATA_TRANSFORMATION_MATRIX = 'transformation_matrix'


def get_data_from_blender(dataset_location: Path, near_boundary: float, far_boundary: float):
    """
    Loads the blender dataset.

    :param dataset_location:    Where the dataset is stored.
    :param near_boundary:       starting distance of the render frustum from the camera.
    :param far_boundary:        farthest distance of the render frustum from the camera.
    :return:        normalized images, Camera to the world matrices, field_of_view in radiance,
                    near_boundary, far_boundary, average_c2w_before_recenter
    """
    images_metadata = _load_metadata(dataset_location)

    transformation_matrices = []
    images = []
    for frame in images_metadata[METADATA_FRAME_METADATA]:
        transformation_matrices.append(frame[METADATA_TRANSFORMATION_MATRIX])
        images.append(imread(dataset_location / frame[METADATA_FILENAME]))
    images, camera_positions = np.asarray(images, dtype=np.float32), np.asarray(transformation_matrices)

    # Center poses on world origin:
    print("Bounds via blender before:", near_boundary, far_boundary)
    print("positions via blender before:", camera_positions[0, :, 3], camera_positions[-1, :, 3])

    camera_positions, average_c2w_before_recenter = recenter_poses(camera_positions)

    # Transform the poses to be in the coordinate space of a normalized sphere:
    bounds = np.array([near_boundary, far_boundary])
    camera_positions, bounds, scale = spherify_poses(camera_positions, bounds)
    near_boundary, far_boundary = bounds[0], bounds[1]

    field_of_view = float(images_metadata[METADATA_FIELD_OF_VIEW])
    print("Bounds via blender after:", float(near_boundary), float(far_boundary))
    print("positions via blender after:", camera_positions[0, :, 3], camera_positions[-1, :, 3])

    return images / 255.0, camera_positions.astype(np.float32), field_of_view,\
           float(near_boundary), float(far_boundary), average_c2w_before_recenter, scale


def get_data_from_colmap(dataset_location: Path):
    """
    Loads the data from a given real world dataset processed by Colmap the same way llff did. Expects to find images and
     a POSES_BOUNDS_NPY file that contains the poses from Colmap.

    :param dataset_location:    Where the dataset is stored.
    :return:       normalized images, Camera to the world matrices, field_of_view in radiance,
                    near_boundary, far_boundary, average_c2w_before_recenter
    """
    images, poses, bds, average_c2w_before_recenter, scale = load_llff_data(dataset_location)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]

    # Push the boundaries of the rendering frustum a bit.
    near = tf.reduce_min(bds) * .9
    far = tf.reduce_max(bds) * 1.0

    h, w, focal = hwf
    field_of_view = np.arctan2(w / 2, focal) * 2

    # Append [0, 0, 0, 1] at the bottom of each pose.
    poses = np.concatenate([poses, np.tile(np.reshape([0, 0, 0, 1], [1, 1, 4]), [poses.shape[0], 1, 1])], -2)
    return images.astype(np.float32), poses.astype(np.float32), \
           float(field_of_view), float(near), float(far), average_c2w_before_recenter, scale


def load_llff_data(path_to_images):
    """
    Loads the data the same way as the llff used.

    :param path_to_images:  Where the dataset is stored.
    :return:    normalized images, Normalized camera to the world matrices, field_of_view in radiance,
                        average_c2w_before_recenter
    """
    loaded_npy_data_from_colmap = np.load(os.path.join(path_to_images, POSES_BOUNDS_NPY))

    # poses_hwf is a rotation matrix, translation vector, and (height, width, focal) column.
    poses_hwf = loaded_npy_data_from_colmap[:, :-2].reshape([-1, 3, 5])  #

    # Currently [-y, x, z]. Will transform to [x, y, z]:
    poses_hwf = poses_hwf[:, :, [1, 0, 2, 3, 4]]
    poses_hwf[:, :, 1] = -poses_hwf[:, :, 1]

    bounds = loaded_npy_data_from_colmap[:, -2:].transpose([1, 0])
    bounds = np.moveaxis(bounds, -1, 0)

    # Center poses on world origin:
    poses_hwf, average_c2w_before_recenter = recenter_poses(poses_hwf)

    # Transform the poses to be in the coordinate space of a normalized sphere:
    poses_hwf, bounds, scale = spherify_poses(poses_hwf, bounds)

    # Load the images:
    images_paths = [os.path.join(path_to_images, image_name) for image_name in sorted(os.listdir(path_to_images)) if
                    image_name.endswith('JPG') or image_name.endswith('jpg') or image_name.endswith('png')]
    images = np.asarray([imread(img_path)[..., :3] / 255. for img_path in images_paths], dtype=np.float32)

    return images, poses_hwf, bounds, average_c2w_before_recenter, scale


def imread(f):
    if str(f).endswith('png'):
        return imageio.imread(f)
    else:
        return imageio.imread(f)


def _load_metadata(dirpath: Path) -> Dict:
    """
    Load metadata from dataset directory.

    :param dirpath:     Path to the json data file.
    :return:    Metadata dict.
    """
    json_path = dirpath / CAM_DATA_JSON_FILE_NAME
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_weights(neural_net: Model, filepath: Union[str, Path]):
    """
    Saves the model of the NeuralNet to the filepath as a h5 file. Will create a folder if it does not exist.

    :param neural_net:  neural net to save weights.
    :param filepath:    Where to save.
    """
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    neural_net.save_weights(filepath)
    print(f"Saved {filepath}!")


def save_psnr_values(psnrs_test_values, psnrs_train_values, filepath: Path):
    """
    Save the PSNR values in the given filepath.

    :param psnrs_test_values:   List like of PSNR values.
    :param psnrs_train_values:  List like of PSNR values.
    :param filepath:            Where to save.
    """
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(str(filepath), (psnrs_test_values, psnrs_train_values))
    print(f"Saved {filepath}!")


def load_config(config_file_path):
    """
    Loads configuration from the given config filepath.

    :param config_file_path:    Where is the config file.
    :return:        Dictionary of configuration values.
    """
    if not os.path.exists(config_file_path):
        raise Exception(f"Config file '{config_file_path}' not found.")

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def get_psnr_values(path_to_existing_psnr_values: Path):
    """
    Get PSNR and starting epoch num from the given path.

    :param path_to_existing_psnr_values:    Where the psnr values are saved
    :return:    List like of test psnr values, List like of train psnr values
    """
    if path_to_existing_psnr_values and os.path.exists(path_to_existing_psnr_values):
        psnrs_test_values, psnrs_train_values = np.load(str(path_to_existing_psnr_values))
        print(f"Loaded {path_to_existing_psnr_values}!")
    else:
        psnrs_test_values, psnrs_train_values = [], []
    return psnrs_test_values, psnrs_train_values


def extract_first_number(string: str) -> int:
    """
    Extract first number in the given string.

    :param string:  String to extract first number from.
    :return:        The number.
    """
    return int(re.findall(r'\d+', string)[0])


def extract_last_number(string: str) -> int:
    """
    Extract last number in the given string.

    :param string:  String to extract first number from.
    :return:        The number.
    """
    return int(re.findall(r'\d+', string)[-1])


def get_new_save_location_incremented(path_to_config_file: Path, config: Dict) -> Path:
    """
    Get new location to save the results of the execution run.
    If the save location in the config file does not exist, it will create the directory.
    else, it will return a new directory with incremented number in the name.

    :param path_to_config_file:     Path to the config file.
    :param config:                  The configuration dictionary.
    :return:    New save location.
    """
    general_save_location = config[GENERAL_SAVE_LOCATION]
    if not os.path.exists(general_save_location):
        os.makedirs(general_save_location)
        max_dir_idx = 1
    else:
        dir_names = []
        regex_format = SAVE_DIR_NAME_FORMAT_REGEX.format(path_to_config_file.stem)
        for file_name in os.listdir(general_save_location):
            if re.match(regex_format, file_name):
                dir_names.append(file_name)
        if not dir_names:
            max_dir_idx = 0
        else:
            max_dir_idx = max([extract_last_number(dir_name) for dir_name in dir_names]) + 1
    general_save_location = Path(general_save_location)
    new_save_dir = general_save_location / SAVE_DIR_NAME_FORMAT.format(str(path_to_config_file.stem), max_dir_idx)
    os.makedirs(new_save_dir)
    print("Created save location: " + str(new_save_dir))
    return new_save_dir


def get_save_location(path_to_config_file, config) -> Path:
    """
    Get save location from config file, or create it if it doesn't exist/specified.
    If is not specified and there are files with that configuration name, will create a new directory with
    incremented number in the name.

    :param path_to_config_file:     Path to the config file.
    :param config:                  Configuration file dictionary.
    :return:        Save location to save results in.
    """
    existing_save_dir_name = config[EXISTING_SAVE_DIR_NAME]
    if existing_save_dir_name:
        general_save_location = config[GENERAL_SAVE_LOCATION]
        existing_save_location = Path(general_save_location) / existing_save_dir_name
        if not os.path.exists(existing_save_location):
            raise Exception('Save location', existing_save_location, 'does not exists.')
        return existing_save_location
    else:
        return get_new_save_location_incremented(path_to_config_file, config)
