import os
import shutil
import time
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Union, List

from cloudpathlib import CloudPath
import subprocess
import numpy as np
import pytz
import tensorflow as tf
from keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from src.ConfigurationKeys import *
from src.ConfigurationKeys import DATASET_LOCATION, STARTING_EPOCH_NUMBER, TASKS_TO_PERFORM, RENDER, TRAINING, VIDEO
from src.DietNeRF import DietNeRF
from src.NeRF import NeRF, MIXED_FLOAT16
from src.UtilsCV import histogram_equalize, get_c2w_matrices_between_2_c2w, poses_avg, \
    change_mats_to_homogeneous, get_l_to_r_c2w_matrices, \
    get_sphere_matrices, get_c2w_matrices_between_2_c2w_with_stretch, estimate_point_of_interest_in_scene, \
    get_rotation_matrix_from_source_to_dest_mats
from src.UtilsFiles import load_config, get_data_from_blender, get_psnr_values, save_weights, \
    save_psnr_values, \
    get_save_location, get_data_from_colmap
from src.UtilsNeuralRadianceField import prepare_ds, get_psnr_for_image, get_num_of_batches
from src.UtilsPlots import show_and_save_img, create_and_save_epoch_plot, FILENAME_RENDER_L_TO_R_RGB_VIDEO, \
    FILENAME_RENDER_DEPTHS_L_TO_R_VIDEO, FILENAME_TRAIN_SET_VIDEO, FILENAME_PLOTS_VIDEO_MP, \
    FILENAME_RENDER_RGB_SPHERE_VIDEO, FILENAME_RENDER_DEPTHS_SPHERE_VIDEO, get_psnr_save_path, \
    save_visualization_of_rendering_between_2_images, FILENAME_RENDER_RGB_PATH_VIDEO, \
    FILENAME_RENDER_DEPTHS_PATH_VIDEO, merge_along_ray_hierarchical_sampling_results, slice_out_rays
from src.UtilsVideo import save_frames_as_video, save_plot_video, \
    save_plots_that_visualize_values_along_rays

# Dataset type:
COLMAP = 'colmap'
BLENDER = 'blender'

# Directory names
DIR_PLOT_ITERATION_IMAGES = 'plot_iteration_images'
DIR_SAVE_VIDEOS = 'video_save'
DIR_SPECIAL_PLOTS = 'special_plots'

# Math constants:
EPS = 1e-8

# Messages:
MSG_DONE_EPOCH = " - Entire epoch {}# took {:.5} seconds, with average iteration time {:.3}. The test PSNR was: {:.5}"


class ExecutionRun:
    """
    This object represents a single execution of running according to one configuration file.
    It expects a configuration file in the constructor, and will follow the instructions in that file.
    The configuration file will also contain hyperparameters for training, rendering.
    """

    def __init__(self, path_to_config_file: Union[str, Path]):
        """
        Initialize the execution object. Will not start running until the method start() is called.

        :param path_to_config_file: Path to the configuration file.
        """
        config = load_config(path_to_config_file)

        self.tasks_to_perform = config[TASKS_TO_PERFORM]

        self.dataset_type = config[DATASET_TYPE]
        self.pics_indices_to_use_in_dataset = config[PICS_INDICES_TO_USE_IN_DATASET] if \
            PICS_INDICES_TO_USE_IN_DATASET in config else None

        images, poses, fov, near, far, average_c2w_before_recenter, scale = self.get_data(config)
        self.images = images
        self.camera_poses = poses
        self.field_of_view = fov
        self.near_boundary = near
        self.far_boundary = far
        self.average_c2w_before_recenter = average_c2w_before_recenter
        self.c2w_scale_parameter = scale

        # In the save location there will be all the files that will be saved in this execution.
        self.save_location = get_save_location(path_to_config_file, config)

        # Makes a copy of the configuration file in the save location.
        shutil.copyfile(path_to_config_file, self.save_location / Path(path_to_config_file).name)

        # If not -1, will be the current epoch number (must have weights and psnrs in the save location that matches).
        self._epoch_number = config[STARTING_EPOCH_NUMBER] if config[STARTING_EPOCH_NUMBER] > 0 else 0
        self.net_config = config[NEURAL_NET]
        self.render_config = config[RENDER]
        self.training_config = config[TRAINING]
        self.video_properties = config[VIDEO]

        if GOOGLE_CLOUD_BUCKET_NAME in config:
            self.gcp_bucket_dest_to_copy_results_to = config[GOOGLE_CLOUD_BUCKET_NAME]
        else:
            self.gcp_bucket_dest_to_copy_results_to = None

        israel_tz = pytz.timezone("Israel")
        self.datetime_execution_start = datetime.now(israel_tz).strftime('%Y-%m-%d_%H-%M-%S')

    def get_data(self, config):
        dataset_location = Path(PureWindowsPath(config[DATASET_LOCATION]))

        if self.dataset_type == BLENDER:
            # Near and far boundaries from the configuration file.
            n_bound, f_bound = config[RENDER][NEAR_DEPTH_RENDER], config[RENDER][FAR_DEPTH_RENDER]

            return get_data_from_blender(dataset_location, n_bound, f_bound)
        elif self.dataset_type == COLMAP:
            return get_data_from_colmap(dataset_location)

    def start(self) -> None:
        """
        Calling this method will start the execution.
        """
        if self.tasks_to_perform[START_TRAINING]:
            self._training()
            self._epoch_number = self.training_config[N_EPOCHS]  # Done all n epochs.
            self.backup_to_gcp()

        if self.tasks_to_perform[RENDER_AND_SAVE_TEST_L_TO_R_VIDEO]:
            self.render_l_to_r_test_video()
            self.backup_to_gcp()

        if self.tasks_to_perform[RENDER_AND_SAVE_TEST_SPHERE_VIDEO]:
            self.render_sphere_test_video_with_net_weights()
            self.backup_to_gcp()

        if self.tasks_to_perform[RENDER_AND_SAVE_TEST_PATH_VIDEO]:
            self.render_path_test_video_with_net_weights()
            self.backup_to_gcp()

        if self.tasks_to_perform[SAVE_DATASET_VIDEO]:
            self.save_dataset_video()
            self.backup_to_gcp()

        if self.tasks_to_perform[SAVE_PLOTS_VIDEO]:
            self.save_plot_video()
            self.backup_to_gcp()

        if self.tasks_to_perform[CREATE_PLOTS_THAT_VISUALIZE_VALUES_ALONG_RAYS]:
            self.create_plots_that_visualize_values_along_rays()
            self.backup_to_gcp()

        if self.tasks_to_perform[CREATE_PLOT_THAT_VISUALIZE_RENDERING_BETWEEN_2_IMAGES]:
            self.create_plot_that_visualize_interpolation_of_images()
            self.backup_to_gcp()

        self.backup_to_gcp()

    def backup_to_gcp(self):
        if self.gcp_bucket_dest_to_copy_results_to:
            save_location = self.save_location
            dest_bucket: str = self.gcp_bucket_dest_to_copy_results_to

            bucket = CloudPath(dest_bucket)

            if "_datetime_" in str(save_location):
                dest = str(bucket / str(save_location))
            else:
                dest = str(bucket / str(save_location)) + "_datetime_" + self.datetime_execution_start
            print(f"Calling: gsutil -m rsync -r {str(save_location)}  {dest}")
            subprocess.run(["gsutil", "-m", "rsync", "-r", str(save_location), dest])
            print(f"Done calling: gsutil -m rsync -r {str(save_location)}  {dest}")

    def _training(self) -> None:
        """
        Will start training the Neural Radiance Field model according to the configuration file. Will save in each
        iteration of training a plot and a weights file.
        """
        # Prepare data variables for training.
        idx_test, train_images, train_cam_matrices = self._get_train_data_from_loaded_dataset()
        print(f"Number of training images: {len(train_images)}")
        show_and_save_img(self.images[idx_test], self.save_location / DIR_PLOT_ITERATION_IMAGES)
        idx_train_to_plot = self.training_config[IDX_TRAIN_IMG_TO_PLOT]

        psnrs_test, psnrs_train = get_psnr_values(get_psnr_save_path(self.save_location, self._epoch_number))

        h, w = self.images[0].shape[0], self.images[0].shape[1]
        n_rays_in_batch = self.net_config[N_RAYS_IN_BATCH_TRAIN]
        ds = prepare_ds(n_rays_in_batch, train_cam_matrices, train_images, self.field_of_view)

        n_batches = get_num_of_batches(n_rays_in_batch, len(train_cam_matrices), h, w)
        model = self.get_nerf()

        # Training loop:
        for epoch_number in range(self._epoch_number + 1, self.training_config[N_EPOCHS] + 1):
            start_time = time.time()
            model.fit(ds, steps_per_epoch=n_batches)
            psnrs_train, psnrs_test = self.create_plots_for_cur_epoch(model, epoch_number,
                                                                      self.images[idx_train_to_plot],
                                                                      self.camera_poses[idx_train_to_plot],
                                                                      self.images[idx_test],
                                                                      self.camera_poses[idx_test],
                                                                      psnrs_train, psnrs_test)
            save_checkpoint_for_training(self.save_location, epoch_number, model, psnrs_train, psnrs_test)
            print_epoch_message(epoch_number, n_batches, start_time, psnrs_test[-1])
            self.backup_to_gcp()

    def _get_train_data_from_loaded_dataset(self):
        """
        :return:    return the
                        - index of the view used for testing.
                        - train images.
                        - train poses.
        """
        idx_test = self.training_config[TEST_IMG_IDX]
        indices_train_img = self.get_train_images_indices(idx_test)
        train_images = self.images[indices_train_img]
        train_cam_matrices = self.camera_poses[indices_train_img]
        return idx_test, train_images, train_cam_matrices

    def get_nerf(self) -> NeRF:
        """
        :return:    a new NeRF model. Will load weights from config if specified.
        """
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy(MIXED_FLOAT16)
        if self.net_config[TYPE_OF_MODEL] == DietNeRF.__name__:
            model, optimizer = self._init_dietnerf()
        else:
            model = NeRF(self.net_config, self.render_config, self.near_boundary, self.far_boundary)
            optimizer = Adam(self.training_config[OPTIMIZER_LR])
        model.compile(optimizer=optimizer)
        path_model_save = NeRF.get_nerf_model_path(self.save_location, self._epoch_number)
        if path_model_save and os.path.exists(path_model_save):
            print("Loaded weights:", path_model_save)
            model.load_weights(path_model_save)
        return model

    def _init_dietnerf(self):
        """
        Init the dietNeRf model.
        :return:    model and optimizer to compile the model with.
        """
        # Calculate how many steps to use consistency loss:
        h, w = self.images[0].shape[0], self.images[0].shape[1]
        n_rays_in_batch = self.net_config[N_RAYS_IN_BATCH_TRAIN]

        _, train_images, train_cam_matrices = self._get_train_data_from_loaded_dataset()

        n_batches = get_num_of_batches(n_rays_in_batch, len(train_images), h, w)
        n_steps_with_consistency_loss = n_batches * (self.training_config[N_EPOCHS] - self._epoch_number)
        n_steps_with_consistency_loss *= DietNeRF.PERCENTAGE_OF_TRAIN_STEPS_WITH_CONSISTENCY_LOSS

        # These parameters are helpful for training:
        estimated_intersection, is_spherical_dataset = estimate_point_of_interest_in_scene(self.camera_poses)
        estimated_intersection = estimated_intersection if is_spherical_dataset else None
        if estimated_intersection is not None:
            rot_mat_to_in_front_of_point_of_interest = np.eye(4)
            rot_mat_to_in_front_of_point_of_interest[:3, :3] = self.camera_poses[self.training_config[TEST_IMG_IDX]][:3, :3]
        else:
            rot_mat_to_in_front_of_point_of_interest = None

        model = DietNeRF(self.net_config, self.render_config, self.near_boundary, self.far_boundary, train_images,
                         train_cam_matrices, self.field_of_view, int(n_steps_with_consistency_loss),
                         estimated_intersection, rot_mat_to_in_front_of_point_of_interest)
        optimizer = Adam(self.training_config[OPTIMIZER_LR])
        optimizer = LossScaleOptimizer(optimizer)
        return model, optimizer

    def create_plots_for_cur_epoch(self,
                                   model: NeRF,
                                   epoch_number: int,
                                   train_image,
                                   train_c2w,
                                   test_image,
                                   test_c2w,
                                   psnrs_train_values,
                                   psnrs_test_values):
        """
        Create plots for the current epoch. Also calculates the PSNR for those plots, and return these values.

        :param model:               NeRF model that is being trained.
        :param epoch_number:        Current epoch number.
        :param train_image:         The target image for image prediction of train_c2w.
        :param train_c2w:           Camera to the world matrix, part of the training dataset.
        :param test_image:          The target image for image prediction of test_c2w.
        :param test_c2w:            Camera to theworld matrix, not part of the training dataset.
        :param psnrs_train_values:  list of psnr values up until now to use for the psnr/epoch plot of the train image.
        :param psnrs_test_values:   list of psnr values up until now to use for the psnr/epoch plot of the test image.
        """
        h, w = test_image.shape[0], test_image.shape[1]
        test_img_render = tf.cast(model.render_image(test_c2w, self.field_of_view, h, w)[0], tf.float32)
        test_psnr = get_psnr_for_image(test_img_render, test_image)

        psnrs_test_values = np.append(psnrs_test_values, test_psnr.numpy())
        print(f" - Iteration: {epoch_number}, PSNR: {test_psnr:.5}.")

        h, w = train_image.shape[0], train_image.shape[1]
        train_img_render = tf.cast(model.render_image(train_c2w, self.field_of_view, h, w)[0], tf.float32)
        train_psnr = get_psnr_for_image(train_img_render, train_image)

        psnrs_train_values = np.append(psnrs_train_values, train_psnr.numpy())

        path_for_plot_iteration_saves = self.save_location / DIR_PLOT_ITERATION_IMAGES
        create_and_save_epoch_plot(epoch_number, path_for_plot_iteration_saves, psnrs_train_values, psnrs_test_values,
                                   test_image, test_img_render, train_image, train_img_render)
        return psnrs_train_values, psnrs_test_values

    def render_l_to_r_test_video(self):
        """
        Render a left to right movement video using the NeRF model relevant to this ExecutionRun.
        """
        description = 'Rendering images for l_to_r video'
        c2w_matrices = self.get_l_to_r_c2w_matrices_to_render()
        filename_rgb = FILENAME_RENDER_L_TO_R_RGB_VIDEO
        filename_depths = FILENAME_RENDER_DEPTHS_L_TO_R_VIDEO

        self.render_video(c2w_matrices, description, filename_rgb, filename_depths)

    def render_video(self,
                     c2w_matrices,
                     process_description: str,
                     filename_rgb: str,
                     filename_depths: str,
                     loops=1):
        """
        Render a video using the NeRF model relevant to this ExecutionRun. The frames of the video will be created using
        the given camera to the world matrices.

        :param c2w_matrices:        Camera to the world matrices that will be used to render the frames.
        :param process_description: Process description for the current rendering.
        :param filename_rgb:        Name for the rgb video file.
        :param filename_depths:     Name for the depths video file.
        :param loops:               Number of times that the video would be looped.
        """
        model = self.get_nerf()
        h = self.images[0].shape[0]
        w = self.images[0].shape[1]
        fps = self.video_properties[FPS_RENDER_VIDEO]
        camera_matrices_to_render = tf.constant(c2w_matrices, tf.float32)

        rgb_images = []
        depths_images = []
        for cam_matrix in tqdm(camera_matrices_to_render, desc=process_description):
            render_result, weights, _, _, _, z_values = model.render_image(cam_matrix, self.field_of_view, h, w)

            # Append rendered frame:
            rgb_images.append(render_result)

            # Append depth frame:
            depth_image = tf.reduce_sum(weights * z_values, axis=-1)
            depth_image_with_improved_visualization = histogram_equalize(depth_image)[0]
            depths_images.append(depth_image_with_improved_visualization)

        # Save rgb frames as video:
        where_to_save_rgb_video = self.save_location / DIR_SAVE_VIDEOS / filename_rgb
        save_frames_as_video(where_to_save_rgb_video, rgb_images * loops, fps)

        # Save depth frames as video:
        where_to_save_depths_video = self.save_location / DIR_SAVE_VIDEOS / filename_depths
        save_frames_as_video(where_to_save_depths_video, depths_images * loops, fps)

    def get_l_to_r_c2w_matrices_to_render(self):
        """
        :return:    Left to right camera to the world matrices for creating l_to_r video.
        """
        seconds = 5
        total_frames = self.video_properties[FPS_RENDER_VIDEO] * seconds
        matrices_to_render = get_l_to_r_c2w_matrices(total_frames)

        estimated_intersection, is_spherical_dataset = estimate_point_of_interest_in_scene(self.camera_poses)

        if is_spherical_dataset:
            t = self.camera_poses[self.training_config[TEST_IMG_IDX]][:3, 3]
            matrices_to_render[:, :3, 3] = t - matrices_to_render[:, :3, 3]
            rotation_mat_of_test_img = self.camera_poses[self.training_config[TEST_IMG_IDX]][:3, :3]
            matrices_to_render[:, :3, :3] = rotation_mat_of_test_img
            return matrices_to_render
        else:
            average_c2w = change_mats_to_homogeneous(poses_avg(self.camera_poses)[..., :4][None])
            matrices_to_render_translated = average_c2w @ matrices_to_render
            return matrices_to_render_translated

    def render_sphere_test_video_with_net_weights(self):
        """
        Render a sphere movement video using the NeRF model relevant to this ExecutionRun.
        """
        description = 'Rendering images for sphere video'
        c2w_matrices = self.get_sphere_c2w_matrices_to_render()
        filename_rgb = FILENAME_RENDER_RGB_SPHERE_VIDEO
        filename_depths = FILENAME_RENDER_DEPTHS_SPHERE_VIDEO

        self.render_video(c2w_matrices, description, filename_rgb, filename_depths)

    def get_sphere_c2w_matrices_to_render(self):
        """
        :return:    Spherical camera to the world matrices to render for creating sphere video.
        """
        seconds = 6
        total_frames = int(self.video_properties[FPS_RENDER_VIDEO] * seconds)
        matrices = get_sphere_matrices(total_frames)
        estimated_intersection, is_spherical_dataset = estimate_point_of_interest_in_scene(self.camera_poses)
        if is_spherical_dataset:
            rotation_mat_of_test_img = self.camera_poses[self.training_config[TEST_IMG_IDX]][:3, :3]
            rotation_mat = get_rotation_matrix_from_source_to_dest_mats(matrices[0, :3, :3], rotation_mat_of_test_img)
            matrices = rotation_mat @ matrices
            matrices[:, :3, 3] += estimated_intersection
        elif self.dataset_type == BLENDER:
            # Because it is left to right scene, the z axis had been centered. So the distance from the object is
            # the previously known z value of the c2w Because Blender scene center is always on the origin (0,0,0)
            scale = self.c2w_scale_parameter
            distance_from_obj = self.average_c2w_before_recenter[2, 3]

            matrices[:, :3, 3] *= scale * distance_from_obj
            push_unit_sphere_by_forward = np.asarray([0, 0, -scale * distance_from_obj])
            matrices[:, :3, 3] += push_unit_sphere_by_forward
        return matrices

    def render_path_test_video_with_net_weights(self):
        """
        Render a sphere movement video using the NeRF model relevant to this ExecutionRun.
        """
        description = 'Rendering images for path video'
        c2w_matrices = self.get_path_c2w_matrices_to_render()
        filename_rgb = FILENAME_RENDER_RGB_PATH_VIDEO
        filename_depths = FILENAME_RENDER_DEPTHS_PATH_VIDEO

        self.render_video(c2w_matrices, description, filename_rgb, filename_depths)

    def get_path_c2w_matrices_to_render(self):
        """
        Create list of c2w matrices to render that are between the 2 given views from the configuration file.

        :return: list like of camera to the world matrices to render.
        """
        seconds = 2
        total_frames = int(self.video_properties[FPS_RENDER_VIDEO] * seconds)
        c2ws = self.camera_poses[self.video_properties[IMG_INDICES_FOR_PATH_VIDEO]]

        c2w_matrices = []
        for c2w1, c2w2 in zip(c2ws[:-1], c2ws[1:]):
            c2w_matrices.extend(get_c2w_matrices_between_2_c2w_with_stretch(c2w1, c2w2, total_frames))
        c2w_matrices.extend(get_c2w_matrices_between_2_c2w_with_stretch(c2ws[-1], c2ws[0], total_frames))

        return np.asarray(c2w_matrices)

    def save_dataset_video(self):
        """
        Save video of all the images given to the ExecutionRun.
        """
        filename = self.save_location / DIR_SAVE_VIDEOS / FILENAME_TRAIN_SET_VIDEO
        _, train_images, train_cam_matrices = self._get_train_data_from_loaded_dataset()
        save_frames_as_video(filename, train_images, self.video_properties[FPS_TRAIN_SET_VIDEO])

    def get_train_images_indices(self, idx_test):
        """
        gets image train indices excluding the test image.

        :param idx_test:    to exclude.
        :return:    list of indices.
        """

        if self.pics_indices_to_use_in_dataset:
            train_indices = set(self.pics_indices_to_use_in_dataset)
            return [n for n in range(len(self.images)) if n != idx_test and n in train_indices]
        else:
            return [n for n in range(len(self.images)) if n != idx_test]

    def save_plot_video(self):
        """
        Saves video showing all the plots that were created in each epoch during the execution run.
        """
        plots_dir_name = self.save_location / DIR_PLOT_ITERATION_IMAGES
        where_to_save_video = self.save_location / DIR_SAVE_VIDEOS / FILENAME_PLOTS_VIDEO_MP
        fps = self.video_properties[FPS_PLOT_VIDEO]

        if os.path.exists(plots_dir_name):
            save_plot_video(fps, plots_dir_name, where_to_save_video)
        else:
            print(f"Could not find: {plots_dir_name}\nSo didn't create the plot video.")


    def create_plots_that_visualize_values_along_rays(self):
        """
        Save special plots that will be created using the NeRF model relevant to this ExecutionRun.
        """
        model = self.get_nerf()

        w, h = self.images[0].shape[1], self.images[0].shape[0]
        img_idx = self.training_config[IDX_TRAIN_IMG_TO_PLOT]
        c2w = self.camera_poses[img_idx]
        render_img, weights, cumprod, alpha, rgb_output, z = model.render_image(c2w, self.field_of_view, h, w)
        n_bins = self.render_config[N_RENDER_SAMPLES_COARSE]

        interesting_rays_coords = np.asarray(
            ((h // 2, w // 2),
             (h // 4, w // 4),
             (h // 4, w // 2))
        )

        # Slice out the interesting rays.
        alpha, cumprod, rgb_output, weights, z = slice_out_rays(interesting_rays_coords, z, weights, cumprod, alpha,
                                                                rgb_output)

        # Because there might be hierarchical sampling in the NeRF model, "binning" close results will allow us to
        # visualize the data along the ray as it advances along the z axis.
        # Make the hierarchical sampling results from the model be uniformly distributed along the ray (z axis).
        alpha_merged, cumprod_merged, rgb_output_merged, weights_merged = merge_along_ray_hierarchical_sampling_results(
            n_bins, z, weights, cumprod, alpha, rgb_output)

        where_to_save = self.save_location / DIR_SPECIAL_PLOTS
        save_plots_that_visualize_values_along_rays(render_img, interesting_rays_coords, weights_merged, cumprod_merged,
                                                    alpha_merged, rgb_output_merged, self._epoch_number, where_to_save)

    def create_plot_that_visualize_interpolation_of_images(self):
        """
        Create a plot that will show how rendered images are interpolated between 2 images.
        """
        img1, img2, rendered_images = self.get_images_for_visualization_between_2_images()
        where_to_save = self.save_location / DIR_SPECIAL_PLOTS
        save_visualization_of_rendering_between_2_images(img1, img2, rendered_images, where_to_save)

    def get_images_for_visualization_between_2_images(self):
        """
        :return:    Images for the visualization plot of images between 2 real images.
        """
        model = self.get_nerf()

        idx_image2 = self.training_config[IDX_TRAIN_IMG_TO_PLOT]
        # Get 2 indices of images that will render images between those 2.
        if idx_image2 == 0:
            idx_image1 = 0
            idx_image2 = 1
        else:
            idx_image1 = idx_image2 - 1
        img1, img2 = self.images[idx_image1], self.images[idx_image2]
        c2w1, c2w2 = self.camera_poses[idx_image1], self.camera_poses[idx_image2]
        w, h = self.images[0].shape[1], self.images[0].shape[0]
        c2w_matrices = get_c2w_matrices_between_2_c2w(c2w1, c2w2)
        rendered_images = []
        for c2w in c2w_matrices:
            rendered_images.append(model.render_image(c2w, self.field_of_view, h, w)[0])
        return img1, img2, rendered_images


def print_epoch_message(epoch_number: int,
                        n_batches_in_epoch: int,
                        start_time: float,
                        test_psnr: List[float]):
    """
    Prints the message for the current epoch.

    :param epoch_number:        Current epoch number.
    :param n_batches_in_epoch:  Number of batches in the epoch.
    :param start_time:          Time when the epoch started.
    :param test_psnr:           PSNR value between the target test image and the rendered image.
    """
    time_start_to_end = time.time() - start_time
    average_iter_time = time_start_to_end / n_batches_in_epoch
    print(MSG_DONE_EPOCH.format(epoch_number, time_start_to_end, average_iter_time, test_psnr))


def save_checkpoint_for_training(save_location: Path,
                                 epoch_number: int,
                                 model: NeRF,
                                 psnrs_train: np.ndarray,
                                 psnrs_test: np.ndarray):
    """
    Save a checkpoint for the training process.

    :param save_location:   Where to save the checkpoint.
    :param epoch_number:    Current epoch number.
    :param model:           Model instance.
    :param psnrs_train:     list of psnr values up until now.
    :param psnrs_test:      list of psnr values up until now.
    """
    where_to_save_model_weights = NeRF.get_nerf_model_path(save_location, epoch_number)
    save_weights(model, where_to_save_model_weights)

    where_to_save_psnr = get_psnr_save_path(save_location, epoch_number)
    save_psnr_values(psnrs_test, psnrs_train, where_to_save_psnr)
