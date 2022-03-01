import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.UtilsPlots import get_plots_filenames, get_generator_of_plots_from_dir, plot_alpha_with_colors_ray, \
    plot_cumprod_of_ray, plot_weights_and_colors_of_ray

RENDERED_IMAGE_JPG_NAME = 'rendered_image.jpg'


def save_frames_as_video(filename: Path, frames: Union[List, np.ndarray], fps: int):
    """
    Save the given list of frames as a video.
    :param filename:    Where to save.
    :param frames:      Frames to save. Expects pixel values between 0 and 1.
    :param fps:         Frame per seconds to use.
    """
    import cv2

    assert len(frames) > 0

    if not os.path.exists(filename.parent):
        print(filename.parent, "didn't exist, so I created it.")
        os.makedirs(filename.parent)

    height = frames[0].shape[0]
    width = frames[0].shape[1]
    writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for frame in tqdm(frames, desc=f"Saving video {str(filename)}", unit="frames"):
        np_round = np.uint8(np.round(frame * 255))
        frame = cv2.cvtColor(np_round, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    print("Saved video in path", filename)


def save_plot_video(fps: int, plots_dir_name: Path, where_to_save_video: Path):
    """
    Save a plots video by using the plots in the given directory.

    :param fps:                 Number of plots in a second.
    :param plots_dir_name:      Where the plots are.
    :param where_to_save_video: Where to save the video.
    """
    plots_filenames = get_plots_filenames(plots_dir_name)
    plots_generator = get_generator_of_plots_from_dir(plots_dir_name, plots_filenames)

    lower_res_plots = []
    for plot in tqdm(plots_generator, "Resizing plots video", total=len(plots_filenames)):
        img_width, img_height = plot.shape[1], plot.shape[0]
        lower_res_factor = 2.5
        lower_res_plot = cv2.resize(plot,
                                    dsize=(int(img_width / lower_res_factor), int(img_height / lower_res_factor)),
                                    interpolation=cv2.INTER_AREA)
        lower_res_plots.append(lower_res_plot)

    save_frames_as_video(where_to_save_video, lower_res_plots, fps)


def save_plots_that_visualize_values_along_rays(render_img, rays_coords, weights, cumprod, alpha, rgb_output, n_epoch,
                                                where_to_save):
    """
    Create plots that show

    :param render_img:      Rendered image result from the NeRF model when rendering an image.
    :param rays_coords:     Ray's coordinates on the image.
    :param weights:         weights results from the NeRF model when rendering an image.
    :param cumprod:         cumprod results from the NeRF model when rendering an image.
    :param alpha:           alpha results from the NeRF model when rendering an image.
    :param rgb_output:      rgb_output results from the NeRF model when rendering an image.
    :param n_epoch:         Number of epochs to write in the title of the plots.
    :param where_to_save:   Where to save the plots.
    """
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    plt.title(f'Rendered Image After {n_epoch} Epochs')
    plt.imshow(render_img)
    plt.savefig(where_to_save / RENDERED_IMAGE_JPG_NAME, dpi=200)
    plt.show()
    for i, ray_coords in enumerate(rays_coords):
        img_patch = render_img[ray_coords[0] - 10:ray_coords[0] + 10, ray_coords[1] - 10:ray_coords[1] + 10]

        alpha_plot_title = f'Ray {ray_coords} along z axis - plot alpha for each segment in the image {RENDERED_IMAGE_JPG_NAME}'
        plot_alpha_with_colors_ray(alpha[i], rgb_output[i], alpha_plot_title, where_to_save,
                                   img_patch, ray_coords)

        cumprod_plot_title = f'Ray {ray_coords} along z axis - plot cumulative product for each segment in ' \
                             f'the image {RENDERED_IMAGE_JPG_NAME}'
        plot_cumprod_of_ray(cumprod[i], cumprod_plot_title, where_to_save,
                            img_patch, ray_coords)

        weights_plot_title = f'Ray {ray_coords} along z axis - plot weights for each segment in the image {RENDERED_IMAGE_JPG_NAME}'
        plot_weights_and_colors_of_ray(weights[i], rgb_output[i], weights_plot_title, where_to_save,
                                       img_patch, ray_coords)
