#########################################################################################################
# This is the configuration keys file.
# These are all the keys that can and need to be defined in a config file.

#########################################################################################################

# General Configurations:

# Origin of the dataset - blender / colmap
DATASET_TYPE = 'dataset_type'

# Saved directory created previously by execution run.
EXISTING_SAVE_DIR_NAME = 'existing_save_dir_name'

# Will be the model number found in the existing save location used in the execution run. Training will start with this
# number, and rendering will use that model. If no existing save location exists, should be -1.
STARTING_EPOCH_NUMBER = 'starting_epoch_number'

# General save location for all execution runs. Preferably should be "Results".
GENERAL_SAVE_LOCATION = 'general_save_location'

# GCP bucket name if training on the cloud.
GOOGLE_CLOUD_BUCKET_NAME = 'google_cloud_bucket_name'

# Location for the dataset directory.
DATASET_LOCATION = 'dataset_location'

# List of pictures to use from the dataset. Will drop the other images, but keep the poses..
PICS_INDICES_TO_USE_IN_DATASET = 'pics_indices_to_use_in_dataset'

#####################
# Tasks to perform: #
#####################
TASKS_TO_PERFORM = 'tasks_to_perform'

# If true, will tell execution run to train.
START_TRAINING = 'start_training'

# If true, will tell execution run to create plots that help visualize rays going into the scene.
CREATE_PLOTS_THAT_VISUALIZE_VALUES_ALONG_RAYS = 'create_plots_that_visualize_values_along_rays'

# If true, will tell execution run to create plots that help visualize how rendered images look between 2 real views.
CREATE_PLOT_THAT_VISUALIZE_RENDERING_BETWEEN_2_IMAGES = 'create_plot_that_visualize_rendering_between_2_images'

# If true, will save the plots created during training as a video.
SAVE_PLOTS_VIDEO = 'save_plots_video'

# If true, will save the dataset as a video.
SAVE_DATASET_VIDEO = 'save_dataset_video'

# If true, will render and save a video of the scene, while the camera moves from left to right.
RENDER_AND_SAVE_TEST_L_TO_R_VIDEO = 'render_and_save_test_left_to_right_video'

# If true, will render and save a video of the scene, while the camera moving in a spherical movement.
RENDER_AND_SAVE_TEST_SPHERE_VIDEO = 'render_and_save_test_sphere_video'

# If true, will render and save a video of the scene, while the camera moves from first image to the second one and so
# on in a specified path. Image indices should be specified.
RENDER_AND_SAVE_TEST_PATH_VIDEO = 'render_and_save_test_path_video'

#################################
# Neural network configuration: #
#################################
NEURAL_NET = 'neural_net'

TYPE_OF_MODEL = 'type_of_model'

# Dimension of the dense hidden layer.
HIDDEN_LAYER_DIM = 'hidden_layer_dim'

# Dimension of the lase dense hidden layer.
LAST_HIDDEN_LAYER_DIM = 'last_hidden_layer_dim'

# Alpha of the leaky ReLU for the dense layer.
LEAKY_RELU_ALPHA = 'leaky_relu_alpha'

# Number of positional encoding dimension for the 3d coordinates input.
N_POS_ENC_DIM_XYZ = 'n_pos_enc_dim_xyz'

# Number of positional encoding dimension for viewing angle input.
N_POS_ENC_VIEW_DIR = 'n_pos_enc_view_dir'

# Number of angles that the net can get as input. 0 means no angles.
# 1 means angle in the horizontal plane (left, right).
# 2 means angle in the horizontal plane, and the vertical plane.
N_ANGLES_FOR_MODEL = 'n_angles_for_model'

# Number of rays that the model can get as input when training. Depends on GPU memory.
N_RAYS_IN_BATCH_TRAIN = 'n_rays_in_batch_train'

# Number of rays that the model can get as input when rendering. Depends on GPU memory.
N_RAYS_IN_BATCH_RENDER = 'n_rays_in_batch_render'

############################
# Rendering configuration: #
############################
RENDER = 'render'

# Number of samples to sample along a ray going into the scene for the coarse sampling stage (Hierarchical sampling).
N_RENDER_SAMPLES_COARSE = 'n_render_samples_coarse'

# Number of samples to sample along a ray going into the scene for the fine sampling stage (Hierarchical sampling).
# If 0, will not use a second Neural network for fine predictions.
N_RENDER_SAMPLES_FINE = 'n_render_samples_fine'

# The model renders the scene by viewing it as a frustum. This number specifies the distance between the camera and the
# end of the frustum.
FAR_DEPTH_RENDER = 'far_depth_render'

# This number specifies the distance between the camera and the start of the frustum.
NEAR_DEPTH_RENDER = 'near_depth_render'

###########################
# Training configuration: #
###########################
TRAINING = 'training'

# Learning rate for training.
OPTIMIZER_LR = 'optimizer_lr'

# Index of the image to use as a training example. Will be used for the epoch plot.
IDX_TRAIN_IMG_TO_PLOT = 'idx_train_img_to_plot'

# Index of the image to use as a test example. Will not be part of the training set, and will be shown in the
# epoch plot.
TEST_IMG_IDX = 'test_img_idx'

# Number of epochs to do in the training stage.
N_EPOCHS = "n_epochs"

########################
# Video configuration: #
########################
VIDEO = 'video'

# FPS for the video showing the training set.
FPS_TRAIN_SET_VIDEO = 'fps_train_set_video'

# FPS for the rendered videos.
FPS_RENDER_VIDEO = 'fps_render_video'

# FPS for video showing the plots.
FPS_PLOT_VIDEO = 'fps_plot_video'

# Image index, will used for creating the video that moves from one image to another.
IMG_INDICES_FOR_PATH_VIDEO = 'img_indices_for_path_video'
