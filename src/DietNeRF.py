from typing import Dict, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.losses import cosine_similarity
from tensorflow.keras import mixed_precision, losses

from src.NeRF import NeRF, MIXED_FLOAT16
from src.UtilsCV import get_z_values, get_z_vals_from_prob_dist_func, interpolation_type_slerp_for_c2w, \
    get_sphere_matrix
from src.UtilsNeuralRadianceField import get_psnr

EMBEDER_URL = "https://tfhub.dev/sayakpaul/vit_b32_fe/1"
EMBEDDER_INPUT_SIZE = 224


class DietNeRF(NeRF):
    """
    Represent a NeRF model that can train on a sparse dataset by using an additional loss that is called consistency
    loss.

    Consistency loss utilize a 3rd party model that its input is an image and output a latent vector that
    encapsulate the representation of the image. By comparing it to randomly generated image, we can regulate the
    learning procedure.
    """

    # The consistency loss will be added in an interval and not in every step:
    K_INTERVAL_SIZE_FOR_CONSISTENCY_LOSS = 13

    CONSISTENCY_LOSS_WEIGHT = 0.1      # Weight of consistency loss during training.

    PERCENTAGE_OF_TRAIN_STEPS_WITH_CONSISTENCY_LOSS = 0.95

    IMG_SIZE_FOR_CS_LOSS = 150
    N_RENDER_SAMPLES_CS_LOSS = 55

    # Embedder model that will embed the image to a latent space representation vector.
    embedder = None     # Lazy init embedder

    def __init__(self,
                 net_config: Dict,
                 render_config: Dict,
                 near_boundary: float,
                 far_boundary: float,
                 target_images: Union[tf.Tensor, np.ndarray],
                 target_camera_poses: Union[tf.Tensor, np.ndarray],
                 field_of_view,
                 max_steps_of_consistency_loss: int = -1,
                 estimated_intersection=None,
                 rot_mat_to_in_front_of_point_of_interest=None
                 ):
        """
        Initialize the model.

        :param net_config:                  From the user config file.
        :param render_config:               From the user config file.
        :param near_boundary:               Near boundary for rendering.
        :param far_boundary:                Far boundary for rendering.
        :param target_images:               Images from the dataset to be used as target reference when calculating the
                                            semantic loss.
        :param target_camera_poses:         Their poses.
        :param field_of_view:               Field of view for rendering.
        :param max_steps_of_consistency_loss:   Max train step to be allowed to use consistency loss.
                                                            Used for training.
        :param estimated_intersection:          Estimated point of intersection for the views at the scene.
                                                            Used for training.
        :param rot_mat_to_in_front_of_point_of_interest:    Pose of camera that looks at the object from the front.
                                                            Used for training.
        """
        super().__init__(net_config, render_config, near_boundary, far_boundary)
        if DietNeRF.embedder is None:
            cur_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy('float32')
            DietNeRF.embedder = tf.keras.Sequential([
                tf.keras.layers.InputLayer((EMBEDDER_INPUT_SIZE, EMBEDDER_INPUT_SIZE, 3)),
                hub.KerasLayer(EMBEDER_URL, trainable=False)
            ])
            tf.keras.mixed_precision.set_global_policy(cur_policy)

        self.net_config = net_config
        self.render_config = render_config

        self.target_images_embedding = DietNeRF.embedder(DietNeRF.embedder_preprocess(target_images))
        self.camera_poses = target_camera_poses
        self.fov = field_of_view
        self.image_height = target_images[0].shape[0]
        self.image_width = target_images[0].shape[1]

        self.max_steps_of_consistency_loss = max_steps_of_consistency_loss
        self.loss_for_rays = losses.MeanSquaredError()

        self.counter = tf.Variable(0, trainable=False, name="counter")
        self._use_consistency_loss = tf.Variable(True, trainable=False, name="_use_consistency_loss")

        self.point_of_interest_in_scene = estimated_intersection
        self.rot_mat_to_in_front_of_point_of_interest = rot_mat_to_in_front_of_point_of_interest
        self.is_spherical_dataset = self.point_of_interest_in_scene is not None

    def get_config(self):
        """
        :return:    configuration for the model.
        """
        return super().get_config() | {
            'net_config': self.net_config,
            'render_config': self.render_config,
            'target_images_embedding': self.target_images_embedding,
            'camera_poses': self.camera_poses,
            'fov': self.fov,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'max_steps_of_consistency_loss': self.max_steps_of_consistency_loss,
            'loss_for_rays': self.loss_for_rays,
            'counter': self.counter,
            '_use_consistency_loss': self._use_consistency_loss,
            'point_of_interest_in_scene': self.point_of_interest_in_scene,
            'is_spherical_dataset': self.is_spherical_dataset
        }

    def train_step(self, data: tf.Tensor) -> Dict:
        """
        The training step of the model.
        Will regularize the training by using the embedded images target representation and generated source
        representation of images from new poses of the scene during training.

        :param data:      rays origin vectors, rays direction vectors, expected rgb values (None, 3).
        :return:    Metric dictionary results.
        """
        self.counter.assign_add(1)

        # Calculate loss:
        rays_orig, rays_dirs, real_rgb = data
        z = get_z_values(self.near_boundary, self.far_boundary, tf.shape(rays_orig)[0], 1,
                         self.n_render_samples_coarse)[:, 0, :]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Trainable variables to watch:
            trainable_variables = self.model_coarse.trainable_variables
            if self.model_fine:
                trainable_variables += self.model_fine.trainable_variables
            tape.watch(trainable_variables)

            # Calculate losses:
            coarse_render, fine_render, loss, loss_for_rays = self._rgb_render_loss(rays_dirs, rays_orig, z, real_rgb)
            cosine_similarity_loss = tf.cond(self.should_use_consistency_loss(),
                                             lambda: self.calc_consistency_loss(),
                                             lambda: tf.cast(0, tf.float32))
            loss += tf.cast(cosine_similarity_loss, tf.float32)

            # Scale loss if mixed_precision:
            if mixed_precision.global_policy().name == MIXED_FLOAT16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)      # Scaling has to be in the GradientTape scope.

        self._apply_gradients_with_loss(loss, scaled_loss, tape, trainable_variables)

        metrics = self._create_metrics(coarse_render, fine_render, real_rgb, cosine_similarity_loss, loss_for_rays,
                                       loss)
        return metrics

    def _rgb_render_loss(self, rays_dirs, rays_orig, z, real_rgb):
        """
        Calculate loss when rendering rgb pixels from the given rays with the model.
        """
        coarse_render, weights_coarse = self.render_rays(self.model_coarse, rays_orig, rays_dirs, z)[:2]
        loss_for_rays = self.loss_for_rays(real_rgb, coarse_render)
        loss = loss_for_rays
        fine_render = None
        if self.model_fine:
            z_from_dist = get_z_vals_from_prob_dist_func(weights_coarse, z, self.n_render_samples_fine)
            fine_render, weights_fine = self.render_rays(self.model_fine, rays_orig, rays_dirs, z_from_dist)[:2]
            loss_for_rays += self.loss_for_rays(real_rgb, fine_render)
            loss += loss_for_rays
        return coarse_render, fine_render, loss, loss_for_rays

    def _create_metrics(self, coarse_render, fine_render, real_rgb, cosine_similarity_loss, loss_for_rays, loss):
        """
        Create metrics for the train step.
        """
        metrics = {"loss": loss, "loss_for_rays": loss_for_rays}

        psnr_coarse = get_psnr(tf.reduce_mean(tf.square(coarse_render - real_rgb)))
        metrics["psnr_coarse"] = psnr_coarse

        if self.model_fine:
            psnr_fine = get_psnr(tf.reduce_mean(tf.square(fine_render - real_rgb)))
            metrics["psnr_fine"] = psnr_fine

        metrics['cosine_similarity_loss'] = cosine_similarity_loss
        metrics['loss'] += tf.cast(cosine_similarity_loss, tf.float32)

        return metrics

    def _apply_gradients_with_loss(self, loss, scaled_loss, tape, trainable_variables):
        """
        Apply gradients with the loss
        """
        if mixed_precision.global_policy().name == MIXED_FLOAT16:
            scaled_gradients = tape.gradient(scaled_loss, trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def calc_consistency_loss(self):
        """
        Calculate semantic consistency loss by sampling a random target image representation in a latent space, and
        compare it to a randomly generated one as source by using the model.

        :return: cosine similarity loss between target and source latent space representation.
        """
        rand_index = np.random.randint(0, len(self.target_images_embedding), 1)
        target_image_embedding = tf.gather(self.target_images_embedding, rand_index)[0]
        pose = self.sample_random_source_pose()
        field_of_view, h, w = self.fov, self.image_height, self.image_width
        rendered_image = self.render_image(pose, field_of_view, DietNeRF.IMG_SIZE_FOR_CS_LOSS,
                                           DietNeRF.IMG_SIZE_FOR_CS_LOSS, self.batch_size_train,
                                           DietNeRF.N_RENDER_SAMPLES_CS_LOSS,
                                           DietNeRF.N_RENDER_SAMPLES_CS_LOSS)[0]
        source_image_embedding = DietNeRF.embedder(DietNeRF.embedder_preprocess(rendered_image[None]))[0]
        cosine_similarity_loss = DietNeRF.CONSISTENCY_LOSS_WEIGHT * DietNeRF.consistency_loss(source_image_embedding,
                                                                                              target_image_embedding)
        return cosine_similarity_loss

    def should_use_consistency_loss(self):
        """
        :return:    True if should use consistency loss, otherwise False.
        """
        # This: (max_steps_of_consistency_loss > 0) => (counter.value() < max_steps_of_consistency_loss)
        # is same as:
        # (Not max_steps_of_consistency_loss <= 0) or (counter.value() < max_steps_of_consistency_loss)
        is_passed_max_steps_with_c_loss = tf.logical_or(self.max_steps_of_consistency_loss <= 0,
                                                        self.counter.value() < self.max_steps_of_consistency_loss)

        is_passed_an_interval = self.counter.value() % DietNeRF.K_INTERVAL_SIZE_FOR_CONSISTENCY_LOSS == 0
        return tf.logical_and(tf.logical_and(is_passed_an_interval, self._use_consistency_loss),
                              is_passed_max_steps_with_c_loss)

    def sample_random_source_pose(self):
        """
        Sample a random c2w matrix, by using 3 randomly chosen c2w matrices from the dataset and interpolate between
        them.

        :return:    Camera to the world matrix.
        """
        if self.is_spherical_dataset:
            radius = np.random.uniform(0.7, 1.1, 1)[0]
            x_rot = np.random.uniform(-90, 0, 1)[0]
            y_rot = np.random.uniform(-180, 180, 1)[0]
            c2w = get_sphere_matrix(radius, x_rot, y_rot, 0)
            c2w = self.rot_mat_to_in_front_of_point_of_interest @ c2w
            c2w[:3, 3] += self.point_of_interest_in_scene
            return c2w
        else:
            random_choice = np.random.choice(len(self.camera_poses), 3, replace=False)
            chosen_camera_poses = tf.gather(self.camera_poses, random_choice)
            alphas = np.random.uniform(0, 1, 2)
            composited_pose1 = interpolation_type_slerp_for_c2w(chosen_camera_poses[0], chosen_camera_poses[1], alphas[0])
            composited_pose2 = interpolation_type_slerp_for_c2w(composited_pose1, chosen_camera_poses[2], alphas[1])
            return composited_pose2

    @staticmethod
    def consistency_loss(embedding_source, embedding_target):
        """
        Calculate the consistency loss.

        :param embedding_source:    vector of source embedding.
        :param embedding_target:    vector of target embedding.
        :return:    positive consistency loss between [0, 1].
        """
        return tf.squeeze((1 + cosine_similarity(embedding_source[None], embedding_target[None])) / 2)

    @staticmethod
    def embedder_preprocess(images):
        """
        Preprocess for embedder vit_b32_fe.
        :param images:  images to preprocess.
        :return:    images.
        """
        return tf.image.resize(images, size=(EMBEDDER_INPUT_SIZE, EMBEDDER_INPUT_SIZE)) * 2 - 1

    def set_use_consistency_loss(self, should_use: bool):
        self._use_consistency_loss.assign(should_use)

    def is_use_consistency_loss(self) -> bool:
        return self._use_consistency_loss.value()
