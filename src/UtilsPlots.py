import os
import re
from pathlib import Path
from typing import List

import numpy as np
from imageio import imread

from matplotlib import pyplot as plt

from src.UtilsFiles import extract_first_number

# Common filenames:
FILENAME_VISUALIZATION_OF_RENDERING_BETWEEN_2_IMAGES = 'visualization_of_rendering_between_2_images.jpg'
PLOT_FILENAME_PATTERN = r'^.*\d+\.jpg$'
FILENAME_TEST_IMG = 'test_img.jpg'
FILENAME_ITERATION_PLOTS = 'train_iteration_plots_{:03}.jpg'
FILENAME_PLOTS_VIDEO_MP = 'plots_video.avi'
FILENAME_TRAIN_SET_VIDEO = 'train_set_video.avi'
FILENAME_RENDER_DEPTHS_L_TO_R_VIDEO = 'render_depths_l_to_r_video.avi'
FILENAME_RENDER_L_TO_R_RGB_VIDEO = 'render_l_to_r_rgb_video.avi'
FILENAME_RENDER_DEPTHS_SPHERE_VIDEO = 'render_depths_sphere_video.avi'
FILENAME_RENDER_DEPTHS_PATH_VIDEO = 'render_depths_path_video.avi'
FILENAME_RENDER_RGB_SPHERE_VIDEO = 'render_rgb_sphere_video.avi'
FILENAME_RENDER_RGB_PATH_VIDEO = 'render_rgb_path_video.avi'
FILENAME_PSNR_FOR_ITER = 'psnrs_train_test_{:03}.npy'

DIRNAME_TO_SAVE_PSNRS = 'saved_test_train_psnrs'

MATPLOTLIB_DPI = 200


def show_and_save_img(image, where_to_save: Path):
    """
    Shows the given image and save it in the given directory.

    :param image:           array-like or PIL image.
    :param where_to_save:   Path where to save.
    """
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    plt.title('Test Image')
    plt.imshow(image)
    plt.savefig(where_to_save / FILENAME_TEST_IMG, dpi=MATPLOTLIB_DPI)
    plt.show()


def plot_ray(title: str, axis_x_name: str, axis_y_name: str, data):
    """
    Plots a ray data visualization.

    :param title:       Title of the plot.
    :param axis_x_name: Axis x name.
    :param axis_y_name: Axis y name.
    :param data:        List like data to plot.
    :return:        pyplot figure and axis.
    """
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(data, zorder=1)
    ax.grid()
    ax.set_title(title.capitalize())
    ax.set_xlabel(axis_x_name)
    ax.set_ylabel(axis_y_name)
    return fig, ax


def plot_alpha_with_colors_ray(alpha_ray, rgb_image_ray, title: str, where_to_save: Path, img_patch, ray_loc):
    """
    Plots the alpha_ray data visualization.

    :param alpha_ray:       List like data to plot.
    :param rgb_image_ray:   List like data of (R, G, B).
    :param title:           Title of the plot.
    :param where_to_save:   Where to save plot.
    :param img_patch:       Patch of the image from which the ray was extracted.
    :param ray_loc:         The location of the ray in the original image, tuple (y, x).
    """
    fig, ax = plot_ray(title, 'Segment #', 'Alpha Value', alpha_ray)
    add_img_patch(fig, img_patch, ray_loc)
    _scatter_rgb_and_save_plot(fig, ax, title, alpha_ray, rgb_image_ray, where_to_save)


def add_img_patch(fig: plt.Figure, img_patch, ray_loc):
    """
    Add the image patch to the plot.

    :param fig:         Figure to add patch to.
    :param img_patch:   The patch to add.
    :param ray_loc:     The location of the ray.
    """
    img_ax = fig.add_axes([0.82, 0.5, 0.2, 0.2], zorder=1)
    img_ax.imshow(img_patch)

    img_ax.set_title("Image Patch")
    img_ax.set_xticks([img_patch.shape[1] // 2])
    img_ax.set_xticklabels([ray_loc[1]])
    img_ax.set_yticks([img_patch.shape[0] // 2])
    img_ax.set_yticklabels([ray_loc[0]])


def plot_weights_and_colors_of_ray(weights_ray, rgb_image_ray, title: str, where_to_save: Path, img_patch, ray_loc):
    """
    Plots the weight's data visualization.

    :param weights_ray:     List like data to plot.
    :param rgb_image_ray:   List like data of (R, G, B).
    :param title:           Title of the plot.
    :param where_to_save:   Where to save the plot.
    :param img_patch:       Patch of the image from which the ray was extracted.
    :param ray_loc:         The location of the ray in the original image, tuple (y, x).
    """
    fig, ax = plot_ray(title, 'Segment #', 'Weight Value', weights_ray)
    add_img_patch(fig, img_patch, ray_loc)
    _scatter_rgb_and_save_plot(fig, ax, title, weights_ray, rgb_image_ray, where_to_save)


def _scatter_rgb_and_save_plot(fig: plt.Figure, ax: plt.Axes, title: str, alpha_ray, rgb_image_ray, where_to_save: Path):
    """
    Scatter the RGB data and save the figure.

    :param fig:             Figure to plot the rgb on and then save.
    :param ax:              Axis of the plot.
    :param title:           Title of the plot.
    :param alpha_ray:       List like data to use for plotting the color.
    :param rgb_image_ray:   List like data to use for plotting the color.
    :param where_to_save:   Where to save the plot.
    """
    ax.scatter(range(len(alpha_ray)), alpha_ray, alpha=alpha_ray, c=rgb_image_ray, zorder=2,
               s=30, linewidths=1, edgecolors='black')
    fig.show()
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    fig.savefig(where_to_save / f'{title}.jpg', dpi=MATPLOTLIB_DPI)


def plot_cumprod_of_ray(cumprod_ray, title: str, where_to_save: Path, img_patch, ray_loc):
    """
    Plots the cummulative product of ray given by the NeRF model.

    :param cumprod_ray:     List like data to plot.
    :param title:           Title of the plot.
    :param where_to_save:   Where to save the plot.
    :param img_patch:       Image patch of the area that the ray was drawn from.
    :param ray_loc:         Location of the ray on the image, (y, x).
    """
    fig, ax = plot_ray(title, 'Segment #', 'Cumprod Value', cumprod_ray)
    add_img_patch(fig, img_patch, ray_loc)
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    fig.savefig(where_to_save / f'{title}.jpg', dpi=MATPLOTLIB_DPI)
    fig.show()


def create_and_save_epoch_plot(epoch_number: int,
                               plots_save_location: Path,
                               psnrs_train_values,
                               psnrs_test_values,
                               test_image,
                               test_img_render,
                               train_image,
                               train_img_render):
    """
    Create the plot that visualize the NeRF rendering after an epoch training.

    :param epoch_number:            Current epoch number.
    :param plots_save_location:     Where the plots are saved.
    :param psnrs_train_values:      List like of psnr values.
    :param psnrs_test_values:       List like of psnr values.
    :param test_image:              Real test image.
    :param test_img_render:         Rendered test image by NeRF.
    :param train_image:             Real train image.
    :param train_img_render:        Rendered train image by NeRF.
    """
    plt.figure(figsize=(16, 10))
    plt.subplot(231)
    plt.title("Test Image (not part of the training set)")
    plt.imshow(test_image)
    plt.subplot(232)
    plt.title(f"Test Render Result, Epoch: {epoch_number}")
    plt.imshow(test_img_render)
    plt.subplot(233)
    plt.title('PSNR For Test Image')
    plt.plot(np.arange(1, len(psnrs_test_values) + 1), psnrs_test_values)
    plt.xlabel("Epoch number")
    plt.ylabel("PSNR value")
    plt.grid()
    plt.subplot(234)
    plt.title("Train Image (part of the training set)")
    plt.imshow(train_image)
    plt.subplot(235)
    plt.title(f"Train Render Result, Epoch: {epoch_number}")
    plt.imshow(train_img_render)
    plt.subplot(236)
    plt.plot(np.arange(1, len(psnrs_train_values) + 1), psnrs_train_values)
    plt.grid()
    plt.xlabel("Epoch number")
    plt.ylabel("PSNR value")
    plt.title('PSNR For Train Image')
    if not os.path.exists(plots_save_location):
        os.makedirs(plots_save_location)
    plt.savefig(plots_save_location / FILENAME_ITERATION_PLOTS.format(epoch_number), dpi=MATPLOTLIB_DPI)
    plt.show()


def get_plots_filenames(path_to_iteration_plots: Path):
    """
    Get all the filenames of plots from the given path.

    :param path_to_iteration_plots:     Where to look for plots.
    :return:        List of filenames of plots.
    """
    plots_filenames = []
    for file_name in os.listdir(path_to_iteration_plots):
        if re.match(PLOT_FILENAME_PATTERN, file_name):
            plots_filenames.append(file_name)
    plots_filenames = sorted(plots_filenames, key=extract_first_number)
    return plots_filenames


def get_generator_of_plots_from_dir(dir_name: Path, plots_filenames: List[str]) -> np.ndarray:
    """
    Return generator of plots from the given directory.

    :param dir_name:            Where the plots are stored.
    :param plots_filenames:     Names of the plots to pull out from the directory.
    :return:        Normalized plot as a numpy array.
    """
    for file_name in plots_filenames:
        image = imread(dir_name / file_name)
        yield np.asarray(image, dtype=np.float32) / 255.


def get_psnr_save_path(save_location: Path, epoch_file_number: int) -> Path:
    """
    Get location to save psnr values.

    :param save_location:       Where to save psnr values.
    :param epoch_file_number:   Current epoch number.
    :return:    Location to save the psnr values.
    """
    return save_location / DIRNAME_TO_SAVE_PSNRS / FILENAME_PSNR_FOR_ITER.format(epoch_file_number)


def save_visualization_of_rendering_between_2_images(img1, img2, rendered_images, where_to_save):
    """
    Plot and save a visualization of rendering between 2 images.

    :param img1:                First real image.
    :param img2:                Second real image.
    :param rendered_images:     The rendered images between img1 and img2.
    :param where_to_save:       Where to save the visualization.
    """
    n_renders = len(rendered_images)

    plt.figure(figsize=(20, 3))
    plt.suptitle('Visualization of interpolation between 2 images')
    plt.subplot(2, n_renders, 1)
    plt.title("Real Image 1")
    plt.axis('off')
    plt.imshow(img1)
    plt.subplot(2, n_renders, n_renders)
    plt.title(f"Real Image 2")
    plt.axis('off')
    plt.imshow(img2)
    for i, img in enumerate(rendered_images):
        plt.subplot(2, n_renders, n_renders + i + 1)
        plt.title("Render " + str(i + 1))
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(top=0.9, bottom=0.05, right=1, left=0,
                        hspace=0.2, wspace=0.1)
    plt.margins(0, 0)
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    plt.savefig(where_to_save / FILENAME_VISUALIZATION_OF_RENDERING_BETWEEN_2_IMAGES, dpi=MATPLOTLIB_DPI)
    plt.show()


def slice_2d_arr_with_start_end_indices(arr, start_indices_2d, end_indices_2d):
    """
    Slice an array using an array of start indices and end indices.


    :param arr:                 array to slice.
    :param start_indices_2d:    indices to start slice from.
    :param end_indices_2d:      matching indices to end of slice.
    :return:        (array with extra dimension where the sliced values are stored in,
                        mask that where true means real sliced value)
    """
    r = np.arange(arr.shape[1])
    mask = (start_indices_2d[..., None] <= r) & (end_indices_2d[..., None] >= r)
    if len(arr.shape) > 2:
        mask_with_missing_dims = np.expand_dims(mask, tuple(np.arange(2, len(arr.shape))+1))
        mask = np.broadcast_to(mask_with_missing_dims, mask.shape + arr.shape[2:])
    arr_with_extra_sliced_dim = np.where(mask, np.broadcast_to(arr[:, None, ...], mask.shape), np.zeros(mask.shape))
    return arr_with_extra_sliced_dim, mask


def slice_along_z_axis(arr, coords_pairs):
    """
    slice an array using pairs of (i1, i2) coordinates.

    :param arr:             Array to slice.
    :param coords_pairs:    Pairs of coordinates.
    :return:    sliced array.
    """
    return np.asarray(arr)[tuple(coords_pairs.T) + np.index_exp[:]]


def merge_along_ray_hierarchical_sampling_results(n_bins, z, weights, cumprod, alpha, rgb_output):
    """
    Merge along ray the results of a hierarchical sampling, thus making the hierarchical sampling results from the
     model be uniformly distributed along the ray (z axis).

    :param n_bins:      Number of bins to merge to.
    :param z:           locations along z axis that the model was asked to sample along.
    :param weights:     weights results from model.
    :param cumprod:     cumprod results from model.
    :param alpha:       alpha results from model.
    :param rgb_output:  rgb_output results from model.
    :return:        the model results merged and uniformly distributed along the z axis.
    """
    hist_results_for_rays = [np.histogram(ray, bins=n_bins) for ray in z]
    count, z_after_hist = list(map(list, zip(*hist_results_for_rays)))  # Transpose data
    count = np.asarray(count).reshape(list((*z.shape[:-1], -1)))
    cumsum_count = np.cumsum(count, axis=-1)

    # concatenate 0 column and drop last column.
    start_indices_2d = np.hstack((np.zeros((cumsum_count.shape[0], 1)), cumsum_count[..., :-1]))
    end_indices_2d = cumsum_count.copy()

    sliced_weights, mask = slice_2d_arr_with_start_end_indices(weights, start_indices_2d, end_indices_2d)
    weights_merged = np.sum(np.ma.masked_where(mask ^ True, sliced_weights), axis=2).tolist()

    sliced_cumprod, mask = slice_2d_arr_with_start_end_indices(cumprod, start_indices_2d, end_indices_2d)
    cumprod_merged = np.product(np.ma.masked_where(mask ^ True, sliced_cumprod), axis=2).tolist()

    sliced_alpha, mask = slice_2d_arr_with_start_end_indices(alpha, start_indices_2d, end_indices_2d)
    alpha_merged = np.sum(np.ma.masked_where(mask ^ True, sliced_alpha), axis=2).clip(0, 1).tolist()

    sliced_rgb_output, mask = slice_2d_arr_with_start_end_indices(rgb_output, start_indices_2d, end_indices_2d)
    rgb_output_merged = np.mean(np.ma.masked_where(mask ^ True, sliced_rgb_output), axis=2).tolist()

    return alpha_merged, cumprod_merged, rgb_output_merged, weights_merged


def slice_out_rays(rays_coords, z, weights, cumprod, alpha, rgb_output):
    """
    Slice the rays that are specified.

    :param rays_coords:     Coordinates of the rays to slice.
    :param z:               locations along z axis that the model was asked to sample along.
    :param weights:         weights results from model.
    :param cumprod:         cumprod results from model.
    :param alpha:           alpha results from model.
    :param rgb_output:      rgb_output results from model.
    :return:        sliced out rays.
    """
    z = slice_along_z_axis(z, rays_coords)
    weights = slice_along_z_axis(weights, rays_coords)
    cumprod = slice_along_z_axis(cumprod, rays_coords)
    alpha = slice_along_z_axis(alpha, rays_coords)
    rgb_output = slice_along_z_axis(rgb_output, rays_coords)
    return alpha, cumprod, rgb_output, weights, z
