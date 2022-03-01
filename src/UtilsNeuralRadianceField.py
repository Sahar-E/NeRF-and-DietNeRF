import math

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.keras import Model

from src.UtilsCV import get_view_directions, get_rays_directions, sample_along_rays

N_RAYS_TO_SHUFFLE_WITH_SKLEARN = 2_000_000

N_COORDINATES = 3
N_COLOR_CHANNELS = 3
XYZ_COORDS = "xyz_coords"
VIEW_DIRS = "view_dirs"


def split_to_batches(to_split, batch_size):
    """
    Splits the given data into batches of given size.

    :param to_split:        Array to split.
    :param batch_size:      Size of each batch.
    :return:        array in batches.
    """
    assert batch_size > 0
    total_len = to_split.shape[0]
    size_of_splits = get_size_of_splits(batch_size, total_len)
    batches = tf.split(to_split, size_of_splits)
    return batches


@tf.function
def get_size_of_splits(batch_size, total_size):
    """
    Return array containing the sizes of each batch when splitting an array of size total_size into batches of
    size batch_size.

    :param batch_size:  size of each batch.
    :param total_size:  the total size of the array.
    :return:    array where each element is the size of each batch.
    """
    n_full_batches = total_size // batch_size
    if n_full_batches == 0:
        size_of_splits = [total_size]
    elif total_size % batch_size != 0:
        size_of_splits = [batch_size] * n_full_batches + [-1]
    else:
        size_of_splits = [batch_size] * n_full_batches
    return size_of_splits


@tf.function
def positional_encoding_for_views(x: tf.Tensor, n_positional_encoding: int):
    """
    Get positional encoding for vector x.
    :param x:                           Vector to encode.
    :param n_positional_encoding:       Dimensionality of encoding.
    :return:    Vector position after applying positional encoding.
    """
    pow_of_2 = tf.cast(tf.pow(2., tf.range(n_positional_encoding, dtype=tf.float32)), x.dtype)
    theta = pow_of_2 * tf.constant(math.pi, x.dtype) * x[..., tf.newaxis]
    sin_res, cos_res = tf.math.sin(theta), tf.math.cos(theta)
    stack_sin_cos = tf.stack((sin_res, cos_res), axis=-1)
    positional_encoded_x = tf.reshape(stack_sin_cos, [-1, tf.reduce_prod(stack_sin_cos.shape[1:])])
    return positional_encoded_x


@tf.function
def positional_encoding_for_xyz(xyz: tf.Tensor, n_positional_encoding: int):
    """
    Get positional encoding for vector position x.
    :param xyz:                           Vector to encode.
    :param n_positional_encoding:       Dimensionality of encoding.
    :return:    Vector position after applying positional encoding.
    """
    if n_positional_encoding == 0:
        return tf.reshape(xyz, [xyz.shape[0], -1])
    pow_of_2 = tf.cast(tf.pow(2., tf.range(n_positional_encoding, dtype=tf.float32)), xyz.dtype)
    theta = pow_of_2 * tf.constant(math.pi, xyz.dtype) * xyz[..., tf.newaxis]
    sin_res, cos_res = tf.math.sin(theta), tf.math.cos(theta)
    stack_sin_cos = tf.stack((sin_res, cos_res), axis=-1)
    positional_encoded_x = tf.reshape(stack_sin_cos, (-1, 3, stack_sin_cos.shape[-1] * stack_sin_cos.shape[-2]))
    concat_x = tf.concat([xyz[..., tf.newaxis], positional_encoded_x], -1)
    flatten_last_dim = tf.reshape(concat_x, [-1, concat_x.shape[-1] * concat_x.shape[-2]])
    return flatten_last_dim


@tf.function
def ray_marching(model_output, z_values):
    """
    March along the model outputs in the z axis, calculating the final color for the pixel in the image plane.

    :param model_output:    rgba outputs for the samples sampled along rays going into the scene.
    :param z_values:        z value in the world space for the samples sampled along rays.
    :return:    rgb image (pixel color) and
                        weights, cumprod, alpha, net_rgb_output,
                        that were computed while calculating the color of the pixel.
    """
    model_output, z_values = tf.cast(model_output, tf.float32), tf.cast(z_values, tf.float32)
    sigma_a = tf.nn.relu(model_output[..., 3])
    net_rgb_output = tf.math.sigmoid(model_output[..., :3])

    # Create delta steps for calculating the integral.
    delta = z_values[..., 1:] - z_values[..., :-1]
    distance_to_inf = tf.broadcast_to(1e9, tf.concat((tf.shape(delta)[:-1], [1, ]), axis=0))
    delta = tf.concat([delta, distance_to_inf], -1)

    # Using Max's: Optical Models for Direct Volume Rendering, definitions.
    # Where t is between start of the frustum and end, r is the location on the ray, and sigma is the volume density.
    # Alpha is approximation to sigma(r(t)) from the article.
    alpha = 1. - tf.exp(- sigma_a * delta)
    cumprod = tf.math.cumprod(1. - alpha, -1, exclusive=True)
    weights = alpha * cumprod
    rgb_image = tf.reduce_sum(weights[..., tf.newaxis] * net_rgb_output, -2)
    return rgb_image, weights, cumprod, alpha, net_rgb_output


def get_psnr_for_image(image_source, image_target):
    loss = tf.reduce_mean(tf.square(image_source - image_target))
    return get_psnr(loss)


def get_psnr(mse):
    """
    Peak signal-to-noise ratio.
    psnr value for values that have max value of 1.

    :param mse:     Mean Square Error.
    :return:        Peak signal-to-noise ratio.
    """
    psnr_val = -10. * tf.math.log(mse) / tf.math.log(tf.constant(10., mse.dtype))
    return psnr_val


def prepare_ds(batch_size, c2w_matrices, images, fov):
    """
    Prepare the dataset for training.

    :param batch_size:      size of dataset batch to use in the resulting dataset.
    :param c2w_matrices:    Camera to world matrices of the images.
    :param images:          Images for the dataset.
    :param fov:             Field of view of images.
    :return:    Dataset that is shuffled
    """
    field_of_view = tf.constant(fov, dtype=tf.float32, shape=len(c2w_matrices))

    shuffled_train_cam_matrices, shuffled_train_images = shuffle(c2w_matrices, images)
    ds = tf.data.Dataset.from_tensor_slices((shuffled_train_cam_matrices, field_of_view, shuffled_train_images))

    # transform ds to (rays_orig, rays_dirs, rgb_pixels):
    ds = ds.map(lambda cam_mat, fov, img: c2w_to_rays_prepare_ds(cam_mat, fov, img),
                num_parallel_calls=tf.data.AUTOTUNE)

    # Reshape dataset for training:
    ds = ds.flat_map(lambda orig, dirs, rgb_pixels: tf.data.Dataset.from_tensor_slices((orig, dirs, rgb_pixels)))

    # Shuffle, batch and tell tf to prefetch data.
    ds = ds.shuffle(N_RAYS_TO_SHUFFLE_WITH_SKLEARN) \
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE) \
        .prefetch(tf.data.AUTOTUNE)

    return ds


def c2w_to_rays_prepare_ds(c2w, field_of_view, img):
    """
    Transform the camera to the world matrices to rays with origin and direction.

    :param c2w:             Camera to the world matrix.
    :param field_of_view:   Field of view of the camera in radians.
    :param img:             Image to reshape along with the rays to pixel for each ray.
    :return:        rays origin, ray direction, real pixel rgb for that ray.
    """
    real_rgb_pixels = tf.reshape(img, (-1, 3))
    rays_dirs = get_rays_directions(img.shape[0], img.shape[1], field_of_view, c2w)
    rays_dirs = tf.reshape(rays_dirs, (-1, 4))
    rays_orig = tf.broadcast_to(c2w[..., :, 3], rays_dirs.shape)
    return rays_orig, rays_dirs, real_rgb_pixels


def render_rays(model: Model,
                rays_orig: tf.Tensor,
                rays_dirs: tf.Tensor,
                z_values: tf.Tensor,
                n_pos_enc_for_xyz: int,
                n_pos_enc_for_angles: int,
                n_angles_for_model: int):
    """
    Render according to the given rays, using the neural network.

    :param model:                   Model to predict with.
    :param rays_orig:               rays origin vectors.
    :param rays_dirs:               rays direction vectors.
    :param z_values:                Values that represent the z-coordinate along the ray that goes into the scene.
    :param n_pos_enc_for_xyz:       Number of positional encoding for location.
    :param n_pos_enc_for_angles:    Number of positional encoding for the angle of the ray that goes into the scene.
    :param n_angles_for_model:      Number of angles that the model expects.
    :return:        render_result   shape: (None, 3),
                    weights         shape: (None, #Samples, 1),
                    cumprod         shape: (None, #Samples, 1),
                    alpha           shape: (None, #Samples, 1),
                    rgb             shape: (None, #Samples, 3)
    """
    coords_3d = sample_along_rays(rays_orig, rays_dirs, z_values)[..., :3]
    view_dirs = None if n_angles_for_model == 0 else get_view_directions(coords_3d, rays_dirs, n_angles_for_model)
    xyz = tf.reshape(coords_3d, (-1, 3))
    predictions = model_predict(model, n_pos_enc_for_angles, n_pos_enc_for_xyz, xyz, view_dirs)
    shape_rgba = tf.concat((tf.shape(coords_3d)[:-1], (N_COLOR_CHANNELS + 1,)), axis=0)
    predictions = tf.reshape(predictions, shape_rgba)
    render_result, weights, cumprod, alpha, rgb = ray_marching(predictions, z_values)
    return render_result, weights, cumprod, alpha, rgb


def model_predict(model: Model,
                  n_enc_phi_theta: int,
                  n_pos_enc_for_xyz: int,
                  xyz: tf.Tensor,
                  view_dirs: tf.Tensor = None):
    """
    Apply the net on the given input.

    :param model:                   Keras model to apply.
    :param xyz:                     xyz coordinates for the input of the model.
    :param view_dirs:               view directions for the input of the model, if relevant.
    :param n_pos_enc_for_xyz:       Number of positional encoding for input of the network
    :param n_enc_phi_theta:         Number of positional encoding for input of the network
    :return:        Result from the net, (R, G, B, Sigma).
    """
    xyz_encoded = positional_encoding_for_xyz(xyz, n_pos_enc_for_xyz)
    if view_dirs is not None:
        dir_encoded = positional_encoding_for_views(view_dirs, n_enc_phi_theta)
        return model({XYZ_COORDS: xyz_encoded, VIEW_DIRS: dir_encoded})
    else:
        return model(xyz_encoded)


def get_num_of_batches(n_rays_in_batch: int, n_c2w_mats: int, h: int, w: int) -> int:
    """
    Calculate number of batches for the given input.

    :param n_rays_in_batch:     number of rays in a batch.
    :param n_c2w_mats:          number of matrices.
    :param h:                   height of the image
    :param w:                   width of the image.
    :return:    number of batches.
    """
    return (n_c2w_mats * h * w) // n_rays_in_batch
