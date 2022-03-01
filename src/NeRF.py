from pathlib import Path
from typing import Dict

import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Concatenate, Activation
from tensorflow.keras import Model, mixed_precision, losses

from src.ConfigurationKeys import N_RENDER_SAMPLES_FINE, N_RENDER_SAMPLES_COARSE, \
    N_POS_ENC_DIM_XYZ, N_POS_ENC_VIEW_DIR, N_ANGLES_FOR_MODEL, LEAKY_RELU_ALPHA, HIDDEN_LAYER_DIM, \
    LAST_HIDDEN_LAYER_DIM, N_RAYS_IN_BATCH_RENDER, N_RAYS_IN_BATCH_TRAIN
from src.UtilsCV import get_z_vals_from_prob_dist_func, get_z_values, get_rays_directions
from src.UtilsNeuralRadianceField import get_psnr, N_COORDINATES, N_COLOR_CHANNELS, XYZ_COORDS, VIEW_DIRS, \
    split_to_batches

MIXED_FLOAT16 = "mixed_float16"

DIRNAME_TO_SAVE_WEIGHTS = 'saved_weights'
NAME_NERF_MODEL_FILE = 'NeRF_model_epoch_{:03}.h5'


class NeRF(Model):
    """
    Class that represents the NeRF model.
    """

    def __init__(self, net_config: Dict, render_config: Dict, near_boundary: float, far_boundary: float):
        """
        Constructor for the NeRF model.

        :param net_config:      configuration for the network from the config file.
        :param render_config:   configuration for the rendering from the config file.
        """
        super(NeRF, self).__init__()
        self.model_coarse = NeRF.init_network(net_config)
        if render_config[N_RENDER_SAMPLES_FINE] > 0:
            self.model_fine = NeRF.init_network(net_config)
        else:
            self.model_fine = None

        self.batch_size_render = net_config[N_RAYS_IN_BATCH_RENDER]
        self.batch_size_train = net_config[N_RAYS_IN_BATCH_TRAIN]

        # The frustum boundaries when rendering:
        self.near_boundary = near_boundary
        self.far_boundary = far_boundary

        self.n_render_samples_coarse = render_config[N_RENDER_SAMPLES_COARSE]
        self.n_render_samples_fine = render_config[N_RENDER_SAMPLES_FINE]

        self.n_pos_enc_dim_xyz = net_config[N_POS_ENC_DIM_XYZ]
        self.n_pos_enc_view_dir = net_config[N_POS_ENC_VIEW_DIR]
        self.n_angles_for_model = net_config[N_ANGLES_FOR_MODEL]

        self.loss_for_rays = losses.MeanSquaredError()
        self.build([(None, 4), (None, 4)])

    @staticmethod
    def init_network(net_config: Dict) -> Model:
        """
        Initialize the network according to the given configurations.

        :param net_config:  Configuration dictionary.
        :return:        Neural network model.
        """
        if net_config[N_ANGLES_FOR_MODEL] > 0:
            net = NeRF.get_network_xyz_and_view_dir(net_config[N_POS_ENC_DIM_XYZ],
                                                    net_config[LEAKY_RELU_ALPHA],
                                                    net_config[HIDDEN_LAYER_DIM],
                                                    net_config[LAST_HIDDEN_LAYER_DIM],
                                                    net_config[N_POS_ENC_VIEW_DIR],
                                                    net_config[N_ANGLES_FOR_MODEL])
        else:
            net = NeRF.get_network_only_xyz(net_config[N_POS_ENC_DIM_XYZ],
                                            net_config[LEAKY_RELU_ALPHA],
                                            net_config[HIDDEN_LAYER_DIM],
                                            net_config[LAST_HIDDEN_LAYER_DIM])
        return net

    def get_config(self):
        """
        :return:    configuration for the model.
        """
        return {
            'coarse_model': self.model_coarse.get_config(),
            'fine_model': self.model_fine.get_config(),
            'near_boundary': self.near_boundary,
            'far_boundary': self.far_boundary,
            'n_render_samples_coarse': self.n_render_samples_coarse,
            'n_render_samples_fine': self.n_render_samples_fine,
            'n_positional_encoding': self.n_pos_enc_dim_xyz,
            'n_enc_phi_theta': self.n_pos_enc_view_dir,
            'n_angles_for_model': self.n_angles_for_model,
        }

    def call(self, inputs: tf.Tensor, training: bool = None, mask: bool = None):
        """
        Render the inputs using the model.

        :param inputs:      rays origin vectors, rays direction vectors.
        :param training:    Doesn't do anything.
        :param mask:        Doesn't do anything.
        :return:        Rendered results of the inputs, (r,g,b,sigma).
        """
        rays_orig, rays_dirs = inputs
        render_result, _, _, _, _, _ = self.render(rays_orig, rays_dirs)
        return render_result

    def render(self, rays_orig: tf.Tensor, rays_dirs: tf.Tensor, n_render_samples_c=None, n_render_samples_f=None):
        """
        Render the inputs using the model.

        :param rays_orig:           rays origin vectors
        :param rays_dirs:           rays direction vectors.
        :param n_render_samples_c:  Use this to specify the number of coarse render samples in this rendering.
                                        If not specified, uses the one specified in the config file.
        :param n_render_samples_f:  Same as above.
        :return:        render_result   shape: (None, 3),
                        weights         shape: (None, #Samples, 1),
                        cumprod         shape: (None, #Samples, 1),
                        alpha           shape: (None, #Samples, 1),
                        rgb             shape: (None, #Samples, 3)
        """
        z_start = self.near_boundary
        z_end = self.far_boundary
        n_samples = n_render_samples_c if n_render_samples_c else self.n_render_samples_coarse
        z = get_z_values(z_start, z_end, tf.shape(rays_orig)[0], 1, n_samples)[:, 0, :]
        render_result, weights, cumprod, alpha, rgb = self.render_rays(self.model_coarse, rays_orig, rays_dirs, z)
        if self.model_fine:
            n_render_samples_fine = n_render_samples_f if n_render_samples_f else self.n_render_samples_fine
            z_from_dist = get_z_vals_from_prob_dist_func(weights, z, n_render_samples_fine)
            z = tf.sort(tf.concat([z_from_dist, z], axis=-1), axis=-1)
            render_result, weights, cumprod, alpha, rgb = self.render_rays(self.model_fine, rays_orig, rays_dirs, z)
        return render_result, weights, cumprod, alpha, rgb, z

    def train_step(self, data: tf.Tensor) -> Dict:
        """

        The training step of the model.

        :param data:      rays origin vectors, rays direction vectors, expected rgb values (None, 3).
        :return:    Metric dictionary results.
        """
        # Calculate loss:
        rays_orig, rays_dirs, real_rgb = data
        z = get_z_values(self.near_boundary, self.far_boundary, tf.shape(rays_orig)[0], 1,
                         self.n_render_samples_coarse)[:, 0, :]
        trainable_variables = self.model_coarse.trainable_variables
        with tf.GradientTape() as tape:
            coarse_render, weights_coarse = self.render_rays(self.model_coarse, rays_orig, rays_dirs, z)[:2]
            loss = self.loss_for_rays(real_rgb,  coarse_render)

            if self.model_fine:
                trainable_variables += self.model_fine.trainable_variables
                z_from_dist = get_z_vals_from_prob_dist_func(weights_coarse, z, self.n_render_samples_fine)
                fine_render, weights_fine = self.render_rays(self.model_fine, rays_orig, rays_dirs, z_from_dist)[:2]
                loss += self.loss_for_rays(real_rgb,  fine_render)

            if mixed_precision.global_policy().name == MIXED_FLOAT16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if mixed_precision.global_policy().name == MIXED_FLOAT16:
            scaled_gradients = tape.gradient(scaled_loss, trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Add metrics:
        metrics = {"loss": loss}

        psnr_coarse = get_psnr(tf.reduce_mean(tf.square(tf.cast(coarse_render, real_rgb.dtype) - real_rgb)))
        metrics["psnr_coarse"] = psnr_coarse

        if self.model_fine:
            psnr_fine = get_psnr(tf.reduce_mean(tf.square(tf.cast(fine_render, real_rgb.dtype) - real_rgb)))
            metrics["psnr_fine"] = psnr_fine
        return metrics

    def render_rays(self, model, rays_orig, rays_dirs, z: tf.Tensor):
        import src
        return src.UtilsNeuralRadianceField.render_rays(model,
                                                        rays_orig,
                                                        rays_dirs,
                                                        z,
                                                        self.n_pos_enc_dim_xyz,
                                                        self.n_pos_enc_view_dir,
                                                        self.n_angles_for_model)

    def render_image(self, c2w, fov, h, w, batch_size_input=None, n_render_samples_c=None, n_render_samples_f=None):
        """
        Render an image using the NeRF model.

        :param c2w:             Camera to the world coordinate system transformation matrix.
        :param fov:             Camera field of view.
        :param h:               Height resolution of the resulting image.
        :param w:               Width resolution of the resulting image.

        :param batch_size_input:    Use this to specify the size of batch to use for each rendering batch.
                                        If not specified, uses the one specified in the config file.
        :param n_render_samples_c:  Use this to specify the number of coarse render samples in this rendering.
                                        If not specified, uses the one specified in the config file.
        :param n_render_samples_f:  Same as above.
        :return:    render_result_image, weights, cumprod, alpha, rgb, z_values.
        """
        # Prepare rays as input for the model from the c2w and fov arguments:
        c2w = tf.cast(c2w, tf.float32)
        rays_dirs = tf.reshape(get_rays_directions(h, w, fov, c2w), (h * w, 4))
        rays_orig = tf.broadcast_to(c2w[:, 3], (h * w, 4))

        # Split input into batches:
        batch_size = batch_size_input if batch_size_input else self.batch_size_render
        rays_orig_batches = split_to_batches(rays_orig, batch_size)
        rays_dirs_batches = split_to_batches(rays_dirs, batch_size)

        # Loop over batches, while calling NeRF model:
        render_result_parts, weights_parts, cumprod_parts, alpha_parts, rgb_parts, z_parts = [], [], [], [], [], []
        for orig_batch, dirs_batch in zip(rays_orig_batches, rays_dirs_batches):
            render_part, weights_part, cumprod_part, alpha_part, rgb_part, z_part = self.render(orig_batch,
                                                                                                dirs_batch,
                                                                                                n_render_samples_c,
                                                                                                n_render_samples_f)
            render_result_parts.append(render_part)
            weights_parts.append(weights_part)
            cumprod_parts.append(cumprod_part)
            alpha_parts.append(alpha_part)
            rgb_parts.append(rgb_part)
            z_parts.append(z_part)

        # Concat the results:
        render_result = tf.concat(render_result_parts, axis=0)
        weights = tf.concat(weights_parts, axis=0)
        cumprod = tf.concat(cumprod_parts, axis=0)
        alpha = tf.concat(alpha_parts, axis=0)
        rgb = tf.concat(rgb_parts, axis=0)
        z = tf.concat(z_parts, axis=0)

        # Reshape the results:
        render_result = tf.reshape(render_result, (h, w, 3))
        weights = tf.reshape(weights, (h, w, -1))
        cumprod = tf.reshape(cumprod, (h, w, -1))
        alpha = tf.reshape(alpha, (h, w, -1))
        rgb = tf.reshape(rgb, (h, w, -1, 3))
        z = tf.reshape(z, (h, w, -1))

        return render_result, weights, cumprod, alpha, rgb, z

    @staticmethod
    def get_network_only_xyz(n_pos_encoding_xyz: int,
                             leaky_relu_alpha: float,
                             hidden_layer_dim: int,
                             last_hidden_dim: int) -> Model:
        """
        Gets a network that accepts only xyz coordinates.

        :param n_pos_encoding_xyz:  Number of positional encoding for input of the network
        :param leaky_relu_alpha:    The alpha parameter of the LeakyReLU.
        :param hidden_layer_dim:    Dimensionality of the main hidden layers.
        :param last_hidden_dim:     Dimensionality of the last hidden layer.
        :return:    network model.
        """

        def dense_full():
            return Dense(hidden_layer_dim, activation=LeakyReLU(leaky_relu_alpha))

        inputs_xyz = Input(shape=[N_COORDINATES + N_COORDINATES * 2 * n_pos_encoding_xyz])
        hidden = dense_full()(inputs_xyz)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)

        concat_layer1 = Concatenate()([inputs_xyz, hidden])
        hidden = dense_full()(concat_layer1)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)
        sigma_out_full_dense = dense_full()(hidden)

        hidden = dense_full()(sigma_out_full_dense)
        hidden = Dense(last_hidden_dim, activation=LeakyReLU(leaky_relu_alpha))(hidden)

        outputs_rgb = Dense(N_COLOR_CHANNELS, activation=None)(hidden)
        outputs_rgb = Activation('linear', dtype='float32')(outputs_rgb)

        outputs_sigma = Dense(1, activation=None)(sigma_out_full_dense)
        outputs_sigma = Activation('linear', dtype='float32')(outputs_sigma)

        model = Model(inputs_xyz, Concatenate()([outputs_rgb, outputs_sigma]))
        return model

    @staticmethod
    def get_network_xyz_and_view_dir(n_pos_encoding_xyz: int,
                                     leaky_relu_alpha: float,
                                     hidden_layer_dim: int,
                                     last_hidden_dim: int,
                                     n_pos_enc_view_dir: int,
                                     n_angles_for_model: int) -> Model:
        """
        Gets a network that accepts xyz coordinates and view directions.

        :param n_pos_encoding_xyz:  Number of positional encoding for input of the network
        :param leaky_relu_alpha:    The alpha parameter of the LeakyReLU.
        :param hidden_layer_dim:    Dimensionality of the main hidden layers.
        :param last_hidden_dim:     Dimensionality of the last hidden layer.
        :param n_pos_enc_view_dir:  Number of positional of the view direction.
        :param n_angles_for_model:  Number of angles that the model takes as input.
        :return:    network model.
        """

        def dense_full():
            return Dense(hidden_layer_dim, activation=LeakyReLU(leaky_relu_alpha))

        dim_xyz = N_COORDINATES + N_COORDINATES * 2 * n_pos_encoding_xyz
        dim_for_z_in_direction_vec = 1
        dim_directions = n_pos_enc_view_dir * 2 * (n_angles_for_model + dim_for_z_in_direction_vec)

        inputs_xyz = Input(shape=[dim_xyz], name=XYZ_COORDS)
        inputs_view_dirs = Input(shape=[dim_directions], name=VIEW_DIRS)

        hidden = dense_full()(inputs_xyz)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)

        concat_layer1 = Concatenate()([inputs_xyz, hidden])
        hidden = dense_full()(concat_layer1)
        hidden = dense_full()(hidden)
        hidden = dense_full()(hidden)
        dense_sigma_out = dense_full()(hidden)

        dense_sigma_out = Concatenate()([dense_sigma_out, inputs_view_dirs])
        hidden = Dense(last_hidden_dim, activation=LeakyReLU(leaky_relu_alpha))(dense_sigma_out)

        outputs_rgb = Dense(N_COLOR_CHANNELS, activation=None)(hidden)
        outputs_rgb = Activation('linear', dtype='float32')(outputs_rgb)

        outputs_sigma = Dense(1, activation=None)(dense_sigma_out)
        outputs_sigma = Activation('linear', dtype='float32')(outputs_sigma)

        model = Model(inputs=[inputs_xyz, inputs_view_dirs], outputs=Concatenate()([outputs_rgb, outputs_sigma]))
        return model

    @staticmethod
    def get_nerf_model_path(save_location: Path, epoch_number: int) -> Path:
        """
        Returns the path to save the model in.

        :param save_location:   General save location.
        :param epoch_number:    Current number of epoch.
        :return:    Path to save the model in.
        """
        return save_location / DIRNAME_TO_SAVE_WEIGHTS / NAME_NERF_MODEL_FILE.format(epoch_number)
