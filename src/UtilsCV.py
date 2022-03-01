from typing import List, Union

import numpy as np
import quaternion
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

from src.ConfigurationKeys import N_ANGLES_FOR_MODEL

#######################################################################################################################
# This is a fix for "AttributeError: 'int' object has no attribute 'value'" that occurred in the function:
# tfg_transformation.quaternion.from_rotation_matrix
# Found solution from: https://github.com/tensorflow/graphics/issues/15
# The library tensorflow_graphics seems to have some problems with this function.

import sys

module = sys.modules['tensorflow_graphics.util.shape']


def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.v1.dimension_value(tensor.shape[axis])


module._get_dim = _get_dim
sys.modules['tensorflow_graphics.util.shape'] = module
#######################################################################################################################

EPS = 1e-7

RGB_TO_YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.596, -0.275, -0.321],
    [0.212, -0.523, 0.311]
])
X_UNIT_VEC = np.asarray([1, 0, 0])
Y_UNIT_VEC = np.asarray([0, 1, 0])


def get_degrees_from_mat(mat: np.ndarray):
    """
    Get degrees from the rotation matrix.
    :param mat:     4X4 matrix.
    :return:        degrees in x, y and z.
    """
    x_deg = np.rad2deg(np.arctan2(mat[..., 2, 1], mat[..., 2, 2]))
    y_deg = np.rad2deg(np.arctan2(-mat[..., 2, 0], np.sqrt(mat[..., 2, 1] ** 2 + mat[..., 2, 2] ** 2)))
    z_deg = np.rad2deg(np.arctan2(mat[..., 1, 0], mat[..., 0, 0]))
    return x_deg, y_deg, z_deg


def get_x_rot_mat(x_deg: float) -> np.ndarray:
    """
    Gets rotation in the axis x.

    :param x_deg:   angle in x (in degrees).
    :return:    Rotation matrix.
    """
    x_rad = np.deg2rad(x_deg)
    return np.asarray([
        [1, 0, 0, 0],
        [0, np.cos(x_rad), -np.sin(x_rad), 0],
        [0, np.sin(x_rad), np.cos(x_rad), 0],
        [0, 0, 0, 1]
    ])


def get_z_rot_mat(z_deg: float) -> np.ndarray:
    """
    Gets rotation in the axis z.

    :param z_deg:   angle in z (in degrees).
    :return:    Rotation matrix.
    """
    z_rad = np.deg2rad(z_deg)
    return np.asarray([
        [np.cos(z_rad), -np.sin(z_rad), 0, 0],
        [np.sin(z_rad), np.cos(z_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def get_y_rot_mat(y_deg: float) -> np.ndarray:
    """
    Gets rotation in the axis y.

    :param y_deg:   angle in y (in degrees).
    :return:    Rotation matrix.
    """
    y_rad = np.deg2rad(y_deg)
    return np.asarray([
        [np.cos(y_rad), 0, -np.sin(y_rad), 0],
        [0, 1, 0, 0],
        [np.sin(y_rad), 0, np.cos(y_rad), 0],
        [0, 0, 0, 1]
    ])


def get_sphere_matrix(radius: float, x_rot: float, y_rot: float, z_rot: float) -> np.ndarray:
    """
    Matrix for a transformation matrix that moves the vector from the center of the world to a location on the sphere
    with a specific radius, at a specific angle, looking at the origin.

    :param radius:      distance from origin.
    :param x_rot:       angle in x (in degrees).
    :param y_rot:       angle in y (in degrees)
    :param z_rot:       angle in z (in degrees).
    :return:        Translation matrix.
    """
    t = np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1]
    ])
    rotx = get_x_rot_mat(x_rot) @ t
    roty = get_y_rot_mat(y_rot) @ rotx
    rotz = get_z_rot_mat(z_rot) @ roty
    return rotz


def get_view_directions(coords_3d: tf.Tensor, rays_dirs: tf.Tensor, n_angles_for_model: int) -> tf.Tensor:
    """
    Get view directions for the neural network.

    :param coords_3d:           xyz coordinates for the network.
    :param rays_dirs:           rays direction that goes into the scene.
    :param n_angles_for_model:  Number of angles for the model.
    :return:    Rays direction ready for processing.
    """
    if n_angles_for_model == 1:
        indices = [0, 2]  # x and z.
    elif n_angles_for_model == 2:
        indices = [0, 1, 2]  # x, y and z.
    else:
        raise Exception(f'{N_ANGLES_FOR_MODEL} should be 1 or 2.')
    rays_new_shape = tf.concat((tf.shape(coords_3d)[:-1], tf.shape(rays_dirs)[-1:]), axis=0)
    rays_dirs = tf.broadcast_to(rays_dirs[..., tf.newaxis, :], rays_new_shape)
    view_dirs = tf.gather(rays_dirs, indices, axis=-1)
    view_dirs = tf.reshape(view_dirs, (-1, (n_angles_for_model + 1)))
    return view_dirs


def get_c2w_matrices_between_2_c2w(c2w1, c2w2, n_renders: int = 16) -> List:
    """
    Get the camera to the world matrices between the two given c2w.
    Using alpha compositing.

    :param c2w1:        First camera to the world matrix.
    :param c2w2:        Second camera to the world matrix
    :param n_renders:   Number of matrices.
    :return:            List of matrices.
    """
    alphas = np.linspace(0, 1, n_renders)
    c2w_matrices = interpolation_type_slerp_for_c2w(c2w1, c2w2, alphas)
    return c2w_matrices


def alpha_compositing(alpha: Union[float, np.ndarray], arr1: np.ndarray, arr2: np.ndarray):
    """
    Perform alpha compositing between 2 ndarray.
    :param alpha:   The alpha to use. Can be a float or 1-D ndarray.
    :param arr1:    Array 1.
    :param arr2:    Array 2.
    :return:        Array/s that have been alpha composited.
    """
    alpha = np.asarray(alpha)
    if alpha.shape != ():
        alpha = alpha.reshape((-1,) + (1,) * len(arr1.shape))
    return arr1 * (1 - alpha) + arr2 * alpha


def interpolation_type_slerp_for_c2w(c2w1: np.ndarray, c2w2: np.ndarray, alpha: Union[float, np.ndarray, tf.Tensor]):
    """
    Interpolate between c2w1 and c2w2 by using alpha as the factor of transition.

    slerp - spherical linear interpolation
    lerp - linear interpolation
    https://en.wikipedia.org/wiki/Slerp

    :param c2w1:    first camera to the world matrix.
    :param c2w2:    second camera to the world matrix.
    :param alpha:   float between 0 and 1. Can be an array of floats.
    :return:
    """

    def interpolation(m1, m2, t):
        m1, m2 = tf.cast(m1, tf.float32), tf.cast(m2, tf.float32)
        r1 = tfg_transformation.quaternion.from_rotation_matrix(m1[:3, :3])
        r2 = tfg_transformation.quaternion.from_rotation_matrix(m2[:3, :3])
        r = tfg_transformation.rotation_matrix_3d.from_quaternion(slerp_rotation_matrix(r1, r2, t))

        t1, t2 = m1[:3, 3], m2[:3, 3]
        t = t1 * (1 - t) + t2 * t
        return change_mats_to_homogeneous_tf(tf.concat((r, t[..., None]), axis=1)[None])[0]

    alpha = np.asarray(alpha)
    if alpha.shape != ():
        # We have a list of alpha:
        interpolations = [interpolation(c2w1, c2w2, a) for a in alpha]
        return interpolations
    else:
        return interpolation(c2w1, c2w2, alpha)


def slerp_rotation_matrix(p0, p1, t):
    """
    Slerp is shorthand for spherical linear interpolation.
    https://en.wikipedia.org/wiki/Slerp

    :param p0:      First point.
    :param p1:      Second point.
    :param t:       parameter between 0 and 1.
    :return:        Interpolated matrix.
    """
    cos_a = tf.tensordot(p0, p1, 1)

    # Prevent the interpolation interpolating via the longer path around the "sphere",
    # as there are "2 ways" going from a to b on a sphere:
    p1, cos_a = tf.cond(cos_a < 0, lambda: (-p1, -cos_a), lambda: (p1, cos_a))

    omega = tf.acos(cos_a)
    sin_omega = tf.sin(omega)
    return tf.sin((1.0 - t) * omega) / sin_omega * p0 + tf.sin(t * omega) / sin_omega * p1


def get_c2w_matrices_between_2_c2w_with_stretch(c2w1, c2w2, n_renders: int, stretch_knob=1) -> List:
    """
    Get the camera to the world matrices between the two given c2w.
    Using alpha compositing, but there is a stretch that more frames will be near c2w2, making the visual effect of
    slowing down before halt.

    :param c2w1:            First camera to the world matrix.
    :param c2w2:            Second camera to the world matrix
    :param n_renders:       Number of matrices.
    :param stretch_knob:    Knob for stretch. Bigger value will cause less stretch.
    :return:            List of matrices.
    """
    alpha = np.linspace(0, 1, n_renders)

    # Stretch the alphas in order to make the video move fast at the beginning, and then slow down at the end.
    stretched_alpha = alpha * (1 / (alpha + 1 + stretch_knob))
    stretched_alpha = (stretched_alpha - stretched_alpha.min()) / (stretched_alpha.max() - stretched_alpha.min())
    c2w_matrices = interpolation_type_slerp_for_c2w(c2w1, c2w2, stretched_alpha)
    return c2w_matrices


def normalize_vectors(x):
    """
    Normalize vectors.
    :param x:   The vectors.
    :return: normalized vectors.
    """
    return x / np.linalg.norm(x, axis=-1)[..., None]


def get_orthonormal_mat_from_2_vecs(z, y):
    """
    Produce an orthonormal basis from 2 vectors in R^3.

    :param z:   First vector.
    :param y:   Second vector.
    :return:    3D Matrix of orthonormal basis.
    """
    vec2 = normalize_vectors(z)
    vec0 = normalize_vectors(np.cross(y, vec2))
    vec1 = normalize_vectors(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2], 1)
    return m


def poses_avg(poses):
    """
    :param poses:   List of poses in 3d.
    :return:        Returns the average of all poses. 3X4 matrix shape.
    """
    t = poses[:, :3, 3].mean(0)
    r3 = poses[:, :3, 2].mean(0)
    r2 = poses[:, :3, 1].mean(0)
    c2w = np.concatenate([get_orthonormal_mat_from_2_vecs(r3, r2), t[:, None]], 1)
    return c2w


def recenter_poses(poses_hwf):
    """
    Recenter the poses around the origin of the world.
    :param poses_hwf:   List of poses.
    :return:    poses after alignment, average_c2w before alignment.
    """
    average_c2w = poses_avg(poses_hwf[:, :3, :4])
    average_c2w = change_mats_to_homogeneous(average_c2w[..., :4][None])[0]
    poses = change_mats_to_homogeneous(poses_hwf[:, :3, :4])
    poses = np.linalg.inv(average_c2w) @ poses
    poses_hwf[:, :3, :4] = poses[:, :3, :]
    return poses_hwf, average_c2w


def change_mats_to_homogeneous(mats):
    """
    Concatenate [0,0,0,1] as a last row for each matrix.

    param mats:    List of 3X4 matrices.
    :return:        List of 4X4 matrices.
    """
    return np.concatenate([mats, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [mats.shape[0], 1, 1])], 1)


def change_mats_to_homogeneous_tf(mats):
    """
    Concatenate [0,0,0,1] as a last row for each matrix.

    param mats:    List of 3X4 matrices.
    :return:        List of 4X4 matrices.
    """
    return tf.concat([mats, tf.tile(tf.reshape(tf.eye(4)[-1, :], [1, 1, 4]), [mats.shape[0], 1, 1])], 1)


def spherify_poses(poses_hwf, bounds):
    poses_reset = poses_hwf[:, :3, :4]

    radius = np.sqrt(np.max(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    scale = 1. / radius
    poses_reset[:, :3, 3] *= scale
    bounds *= scale
    poses_hwf[:, :3, :4] = poses_reset[:, :3, :4]

    return poses_hwf, bounds, scale


def estimate_intersection_between_lines(dirs_and_t):
    """
    Estimates the intersection between the given lines that are represented as pairs of directions and points on the
    line.

    For more details, please look at https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#General_derivation

    :param dirs_and_t:  pairs of directions and points.
    :return:            estimated intersection using least square.
    """
    if dirs_and_t.shape[0] == 1:
        return None  # There is no intersection with input of 1 line.

    dirs = normalize_vectors(dirs_and_t[:, 0])
    t = dirs_and_t[:, 1]

    I = np.eye(dirs.shape[-1])
    dir_vec_squared = dirs[..., None] @ dirs[..., None, :]
    project_t_and_p_to_space_orth_to_dir_vec = I - dir_vec_squared

    left_mat = np.concatenate(project_t_and_p_to_space_orth_to_dir_vec, axis=0)
    right_vec = np.concatenate(np.squeeze(project_t_and_p_to_space_orth_to_dir_vec @ t[..., None]), axis=0)
    return np.linalg.lstsq(left_mat, right_vec, rcond=None)[0]


def get_distance_of_point_from_line(point, dirs_and_t):
    """
    Gets the distance between the given point and the given lines.

    For more details, please look at https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation

    :param point:       The point.
    :param dirs_and_t:  Lines in the form of direction vector and location vector t on that line.
    :return:    Distances between the point and the lines.
    """
    dirs = normalize_vectors(dirs_and_t[:, 0])
    t = dirs_and_t[:, 1]

    I = np.eye(dirs.shape[-1])
    dir_vec_squared = dirs[..., None] @ dirs[..., None, :]
    project_t_and_p_to_space_orth_to_dir_vec = I - dir_vec_squared

    return np.squeeze((t - point)[..., None, :] @ project_t_and_p_to_space_orth_to_dir_vec @ (t - point)[..., None])


def ransac_get_estimation_for_intersection_point(dirs_and_t, num_iter=10000, inlier_tol=0.001, n_lines=2):
    """
    Perform ransac for intersection estimation of all the lines given.

    :param dirs_and_t:      Lines in the form of direction vector and location vector t on those lines.
    :param num_iter:        Number of iterations to perform.
    :param inlier_tol:      Tolerance to classify a line as an intersection with the estimation of the point.
    :param n_lines:         Number of lines to use when computing an intersection
    :return:            Intersection point estimation, indices of inliers.
    """
    best_n_matches, best_idx = float('-inf'), None
    for _ in range(num_iter):
        random_choice = np.random.choice(dirs_and_t.shape[0], n_lines, replace=False)
        chosen_dirs_and_t = dirs_and_t[random_choice]
        intersection_point = estimate_intersection_between_lines(chosen_dirs_and_t)
        distances = get_distance_of_point_from_line(intersection_point, dirs_and_t)
        is_inlier = distances < inlier_tol
        n_matches = is_inlier.sum()
        if n_matches > best_n_matches:
            best_n_matches = n_matches
            best_idx = np.where(is_inlier)[0]
    if best_n_matches > 1:
        intersection_point_res = estimate_intersection_between_lines(dirs_and_t[best_idx])
        distance = get_distance_of_point_from_line(intersection_point_res, dirs_and_t)
        return intersection_point_res, np.where(distance < inlier_tol)[0]
    else:
        return None, None  # There is no good intersection estimation with only 1 best fitting line.


def get_l_to_r_c2w_matrices(total_frames: int) -> np.ndarray:
    """
    Return list of camera to the world matrices, that rendering with them will create a left to right movement video
    that looks at the scene.

    :param total_frames:    Number of frames to render.
    :return:    List of matrices to render that looks forward at the scene.
    """
    camera_matrices_to_render = []
    x_locations = np.linspace(0, 1, total_frames) * 2 - 1
    for x in x_locations:
        translation_matrix = np.asarray([
            [1., 0., 0., x],
            [0., 1, 0, 0],
            [0., 0, 1, 0],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        camera_matrices_to_render.append(translation_matrix)
    return np.asarray(camera_matrices_to_render)


def get_sphere_matrices(total_n_matrices: int) -> np.ndarray:
    """
    Return the sphere camera to the world matrices for sphere video.

    :param total_n_matrices:    total number of matrices to render.
    :return:    Camera matrices for sphere video.
    """
    matrices = [get_sphere_matrix(1, 0, deg_y, 0) for deg_y in np.linspace(0, 360, total_n_matrices)] + \
               [get_sphere_matrix(1, deg_x, 0, 0) for deg_x in np.linspace(0, 360, total_n_matrices)]
    return np.asarray(matrices, dtype=np.float32)


def estimate_point_of_interest_in_scene(c2w_matrices):
    """
    Estimate the point of interest in the scene by computing an estimation of the intersection of the cameras view
    of the scene.
    Using Ransac and least squares approximation of the point of interest.

    Also estimates if the scene is spherically taken, it is if more than half of the views taken looks at the point
    of interest.
    :param c2w_matrices:    Camera to the world matrices to use for estimation.
    :return:    Estimated point of interest, boolean indicating whether the scene is spherically taken.
    """
    assert len(c2w_matrices) > 1

    dirs_and_t = []
    for c2w in c2w_matrices:
        direction = get_camera_dir_from_c2w(c2w)
        location_t = c2w[:3, 3]
        dirs_and_t.append([direction, location_t])
    dirs_and_t = np.asarray(dirs_and_t)
    estimated_intersection, inliers = ransac_get_estimation_for_intersection_point(dirs_and_t)
    if estimated_intersection is not None and inliers is not None:
        is_spherical_dataset = inliers.shape[0] > 0.3 * dirs_and_t.shape[0]
        return estimated_intersection, is_spherical_dataset
    else:
        return None, False


@tf.function
def get_rays_directions(height, width, field_of_view, c2w):
    """
    Gets the directions vectors for each ray that origins from the camera into the world in world coordinates.
    :param height:          height of the image plane (measured in pixels).
    :param width:           width of the image plane (measured in pixels).
    :param field_of_view:   field of view of the camera.
    :param c2w:             camera to the world matrix.
    :return:    ray direction for each pixel into the world.
    """
    x_raster, y_raster = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))
    # Place coords in the center of the pixel
    x_raster += 0.5
    y_raster += 0.5
    # Transfer from raster space to NDC space:
    x_ndc = x_raster / width
    y_ndc = y_raster / height
    # Transfer to Screen Space:
    x_screen_space = 2 * x_ndc - 1
    y_screen_space = 1 - 2 * y_ndc

    tan_half_fov = tf.tan(field_of_view / 2)
    x_pixel_camera = x_screen_space * tan_half_fov
    y_pixel_camera = y_screen_space * tan_half_fov
    z = -tf.ones_like(x_raster)
    direction_vectors = tf.stack([x_pixel_camera,
                                  y_pixel_camera,
                                  z,
                                  tf.zeros_like(x_raster)], axis=-1)

    # Apply the camera matrix for each direction:
    direction_vectors = tf.einsum('ij,...j', tf.cast(c2w, tf.float32), direction_vectors)
    return direction_vectors


def get_z_vals_from_prob_dist_func(weights, z_values, num_new_z_values):
    """
    Return new  z values along z axis by using the weights as probability distribution.
    This function performs Inverse Transform Sampling.

    :param weights:             weight that will be used as "probability" for a z value.
    :param z_values:            the values along the z axis.
    :param num_new_z_values:    number of samples to sample from the distribution function.
    :return:    new z values sampled along the z axis.
    """
    weights, z_values = tf.cast(weights, tf.float32), tf.cast(z_values, tf.float32)

    probability_dist_func = weights / (tf.reduce_sum(weights, axis=-1, keepdims=True) + EPS)
    cumulative_dist_func = tf.cumsum(probability_dist_func, axis=-1)
    uniform_values = tf.random.uniform(tf.concat((tf.shape(weights)[:-1], [num_new_z_values, ]), axis=0))
    idx_sorted = tf.searchsorted(cumulative_dist_func, uniform_values)

    bottom_range = tf.maximum(0, idx_sorted - 1)
    max_possible_idx = cumulative_dist_func.shape[-1] - 1

    top_range = tf.minimum(max_possible_idx, idx_sorted)
    idx_range = tf.stack([bottom_range, top_range], -1)

    cdf_range = tf.gather(cumulative_dist_func, idx_range, axis=-1, batch_dims=len(idx_range.shape) - 2)

    average_z_values = .5 * (z_values[..., 1:] + z_values[..., :-1])
    indices_clipped = tf.clip_by_value(idx_range, 0, average_z_values.shape[-1] - 1)
    z_range_values = tf.gather(average_z_values, indices_clipped, axis=-1, batch_dims=len(idx_range.shape) - 2)

    # Linear interpolation:
    denominator = (cdf_range[..., 1] - cdf_range[..., 0])
    denominator = tf.where(denominator < 1e-5, 1e-5, denominator)

    t = (uniform_values - cdf_range[..., 0]) / denominator
    z_samples_from_dist = z_range_values[..., 0] + t * (z_range_values[..., 1] - z_range_values[..., 0])
    z_samples_from_dist = tf.sort(z_samples_from_dist, axis=-1)

    return z_samples_from_dist


def transform_to_ndc_space(origin, rays_directions, img_height, img_width, focal_length):
    # Move origin to the near plane at z=-n.
    near = 1.
    t_n = - (near + origin[..., 2]) / rays_directions[..., 2]
    origin = origin + t_n[..., tf.newaxis] * rays_directions

    ox_oz = (origin[..., 0] / origin[..., 2])
    oy_oz = (origin[..., 1] / origin[..., 2])

    origin_x = - (focal_length / (img_width / 2)) * ox_oz
    origin_y = - (focal_length / (img_height / 2)) * oy_oz
    origin_z = 1 + 2 * near / origin[..., 2]
    d = rays_directions
    directions_x = - (focal_length / (img_width / 2)) * (d[..., 0] / d[..., 2] - ox_oz)
    directions_y = - (focal_length / (img_height / 2)) * (d[..., 1] / d[..., 2] - oy_oz)

    directions_z = -2 * near / origin[..., 2]

    origin = tf.stack([origin_x, origin_y, origin_z], axis=-1)
    rays_directions = tf.stack([directions_x, directions_y, directions_z], axis=-1)
    return origin, rays_directions


@tf.function
def get_z_values(z_start, z_end, height, width, n_samples):
    """
    Return z values that are uniformly distributed along a ray (the ray is split into uniform bins, and in each bin the
    value is uniformly distributed).

    :param z_start:     starting value to sample from.
    :param z_end:       final value to sample from.
    :param height:      height of the resulting array.
    :param width:       width of the resulting array.
    :param n_samples:   number of samples to sample.
    :return:    z values in form of 3d array.
    """
    z_values = tf.linspace(z_start, z_end, n_samples)
    z_values = tf.broadcast_to(z_values, (height, width, n_samples))
    z_values += tf.random.uniform([height, width, n_samples]) * (z_end - z_start) / n_samples
    return z_values


@tf.function
def sample_along_rays(origin, direction_vectors, z_values):
    """
    Samples along a ray along that is represented by origin and direction vector.
    The samples will be the given z values.

    :param origin:              origin of the ray.
    :param direction_vectors:   direction vector of the ray.
    :param z_values:            distance to advance in the given direction.
    :return:            coordinates of the samples.
    """
    x = origin[..., tf.newaxis, :]
    d = direction_vectors[..., tf.newaxis, :]
    z_val = z_values[..., tf.newaxis]
    samples_coords = x + d * z_val
    return samples_coords


def get_camera_dir_from_c2w(c2w):
    """
    The camera is looking in the direction of -z.

    :param c2w:  Camera to the world matrix.
    :return:    vector pointing in the camera direction.
    """
    return normalize_vectors(-c2w[:3, 2])


def get_rotation_quaternion_with_axis_vec_and_theta(axis_vec: np.ndarray, theta: float) -> np.ndarray:
    """
    Returns a rotation around the axis vector by about the angle theta.

    :param axis_vec:    3d direction vector.
    :param theta:       Angle in radians.
    :return:            rotation in quaternion.
    """
    real_part = np.cos(theta / 2)[None]
    complex_part = axis_vec * np.sin(theta / 2)
    q = np.concatenate((real_part, complex_part))
    return q


def get_rotation_quaternion_from_vec1_to_vec2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get the rotation in quaternion that is needed to be applied to vec1 in order to rotate it to vec2.

    q = result of this function.
    v2 = q * v1 * q^-1

    For more information, please see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.

    :param v1:  First vector.
    :param v2:  Second vector.
    :return:    rotation in quaternion.
    """
    norm_v1 = normalize_vectors(v1)
    norm_v2 = normalize_vectors(v2)
    dot_v1_v2 = norm_v1.dot(norm_v2)
    if dot_v1_v2 < -0.99999:
        # The vectors are in opposite directions.
        temp_cross = np.cross(X_UNIT_VEC, norm_v1)
        if np.linalg.norm(temp_cross) < 0.00001:
            # They are also parallel to the x-axis.
            temp_cross = np.cross(Y_UNIT_VEC, norm_v1)
        temp_cross = normalize_vectors(temp_cross)
        return get_rotation_quaternion_with_axis_vec_and_theta(temp_cross, np.pi)
    elif dot_v1_v2 > 0.99999:
        # The vectors have the same direction. No rotation needed.
        return np.asarray([1, 0, 0, 0])
    else:
        normalized_cross_v1_v2 = normalize_vectors(np.cross(norm_v1, norm_v2))
        theta = np.arccos(dot_v1_v2)
        return get_rotation_quaternion_with_axis_vec_and_theta(normalized_cross_v1_v2, theta)


def rotate_vec_with_quaternion(vec: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Apply a rotation transformation to vector using the quaternion.

    For more information, please see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation.
    :param vec: 3d vector.
    :param q:   quaternion.
    :return:    3d vector after rotation. Will do q * vec * q^-1
    """
    q_inv = np.quaternion(q[0], -q[1], -q[2], -q[3])
    return quaternion.as_float_array(np.quaternion(*q) * np.quaternion(*vec) * q_inv)[1:]


def get_rotation_matrix_from_v1_to_v2(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get rotation matrix transformation from v1 to v2.

    :param v1:  First vector.
    :param v2:  Second vector.
    :return:    3x3 rotation matrix.
    """
    return quaternion.as_rotation_matrix(np.quaternion(*get_rotation_quaternion_from_vec1_to_vec2(v1, v2)))


def get_rotation_matrix_from_source_to_dest_mats(source_mat: np.ndarray, dest_mat: np.ndarray) -> np.ndarray:
    """
    Gets the rotation matrix from source rotation matrix to dest rotation matrix.
    Uses quaternion, q_rot = q_dest * q_source_inv.

    :param source_mat:  First rotation matrix.
    :param dest_mat:    Second rotation matrix.
    :return:    Rotation matrix transformation from first to second.
    """
    q_dest = quaternion.from_rotation_matrix(dest_mat)
    q_source = quaternion.from_rotation_matrix(source_mat)
    q_rot = q_dest * q_source.inverse()
    rotation_mat = np.eye(4)
    rotation_mat[:3, :3] = quaternion.as_rotation_matrix(q_rot)
    return rotation_mat


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig:     Input grayscale or RGB float64 image with values in [0, 1].
    :return:            The function returns a list [im_eq, hist_orig, hist_eq] where
            im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    im_copy = im_orig[:]
    if len(im_copy.shape) == 3:
        im_yiq = rgb2yiq(im_copy)
        # Calculate the data.
        y = im_yiq[:, :, 0]
        im_eq, hist_orig, hist_eq = _histogram_equalize_grays(y)
        # Restore the RGB format.
        im_yiq[:, :, 0] = im_eq / 255
        return yiq2rgb(im_yiq), hist_orig, hist_eq

    im_eq, hist_orig, hist_eq = _histogram_equalize_grays(im_copy)
    im_eq /= 255
    return im_eq, hist_orig, hist_eq


def _histogram_equalize_grays(gray_im):
    """
    helper method for histogram_equalize that handle gray images. gets pixels in range [0,1]
    and returns in range [0,255].
    """
    if np.max(gray_im) == 0:
        # Cannot histogram_equalize an all zero image.
        return gray_im, None, None
    # Stretch for the histogram
    gray_im -= np.min(gray_im)
    gray_im /= np.max(gray_im)
    gray_im = gray_im * 255
    hist_orig = np.histogram(gray_im, np.arange(257))[0]
    cum_hist = np.cumsum(hist_orig)
    # Normalize:
    nonzero_val = cum_hist[np.nonzero(cum_hist)[0][0]]
    lookup_table = np.round(((cum_hist - nonzero_val) / (cum_hist[-1] - nonzero_val)) * 255)
    im_eq = lookup_table[np.round(gray_im).astype('int')]
    hist_eq = np.histogram(im_eq, np.arange(257))[0]
    return im_eq, hist_orig, hist_eq


def rgb2yiq(im_rgb: np.ndarray) -> np.ndarray:
    """
    Transform the given image from RGB to YIQ and return it.
    :param im_rgb: image.
    """
    return np.dot(im_rgb, RGB_TO_YIQ.T)


def yiq2rgb(im_yiq: np.ndarray) -> np.ndarray:
    """
    Transform the given image from YIQ to RGB and return it.
    :param im_yiq: image.
    """
    yiq_to_rgb_mat = np.linalg.inv(RGB_TO_YIQ)
    return np.dot(im_yiq, yiq_to_rgb_mat.T)
