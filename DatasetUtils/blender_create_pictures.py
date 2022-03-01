"""
Example script that creates blender images for a model that is placed in the origin of the world, and the coordinate's
system of the world is a right-hand one.
"""

import json
import os
import bpy
import mathutils
import math
from time import sleep
import numpy as np
import bpy
from math import *
from mathutils import *

TRANSFORMATION_MATRIX = "transformation_matrix"
FILENAME = "filename"
FRAMES = "frames"
FOCAL_LENGTH = "focal_length"
FIELD_OF_VIEW = "field_of_view"
PROJECT_DIRPATH =  # TODO: change to path of where to save

VIDEO_SPHERE_RADIUS_ROBOT = 3
VIDEO_Z_CLOSEST_DISTANCE_ROBOT = 3
VIDEO_TOTAL_X_DISTANCE_L_TO_R_ROBOT = 3.2
IMG_NAME_FORMAT_ROBOT = "robot_{:02}.png"
CAMERA_ROBOT = 'Camera.Camboto'

# VIDEO_TOTAL_X_DISTANCE_L_TO_R

N_PICS_AND_RESOLUTION_L_TO_R = [
    (8, 100),
    (16, 512),
    (8, 512),
    (16, 256),
    (8, 256),
    (16, 100),
]

N_PICS_AND_RESOLUTION_SPHERE = [
    (72, 512),
    (36, 512),
    (72, 256),
    (36, 256),
    (72, 100),
    (36, 100),
]

N_PICS_AND_RESOLUTION_FULL_CIRCLE = [
    (24, 512),
    (12, 512),
    (24, 256),
    (12, 256),
    (24, 100),
    (12, 100),
]


def change_camera_loc(mat, camera_name):
    cam = bpy.data.objects[camera_name]
    cam.matrix_world = mat


def render_and_save(filepath, w=0, h=0):
    bpy.context.scene.render.filepath = filepath
    if w and h:
        bpy.context.scene.render.resolution_x = w
        bpy.context.scene.render.resolution_y = h
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.render.image_settings.compression = 100
        bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def generate_l_to_r(save_location, n_pics, resolution, camera_name, image_name_format, total_distance_l_to_r,
                    z_closest_distance):
    matrices = []
    for i in range(n_pics):
        x = -(total_distance_l_to_r / 2) + i * (total_distance_l_to_r / (n_pics - 1))
        loc_mat = mathutils.Matrix.Rotation(math.radians(-45), 4, 'X')
        loc_mat = mathutils.Matrix.Translation((x, z_closest_distance, z_closest_distance)) @ loc_mat
        matrices.append(loc_mat)
    render_and_save_dir_matrices(matrices, resolution, save_location, camera_name, image_name_format)


def render_sphere_images(n_pics, resolution, save_location, x_degrees, sphere_radius, camera_name,
                         image_name_format):
    matrices = []
    for rot_x in x_degrees:
        for rot_y in np.linspace(-90, 270, (n_pics // len(x_degrees)) + 1)[:-1]:
            mat = generate_sphere_mat(sphere_radius, rot_x, rot_y, 0)
            matrices.append(mat)
    render_and_save_dir_matrices(matrices, resolution, save_location, camera_name, image_name_format)


def render_full_circle(n_pics, resolution, save_location, sphere_radius, camera_name,
                       image_name_format):
    matrices = []
    for rot_y in np.linspace(-90, 270, n_pics + 1)[:-1]:
        mat = generate_sphere_mat(sphere_radius, 0, rot_y, 0)
        matrices.append(mat)
    render_and_save_dir_matrices(matrices, resolution, save_location, camera_name, image_name_format)


def save_json(dirpath, cam_data):
    json_filepath = os.path.join(dirpath, 'cam_data.json')
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(cam_data, f, ensure_ascii=False, indent=4)


def generate_sphere_mat(radius, x_rot, y_rot, z_rot):
    t = mathutils.Matrix.Translation((0, 0, radius))
    rotx = mathutils.Matrix.Rotation(math.radians(x_rot), 4, 'X') @ t
    roty = mathutils.Matrix.Rotation(math.radians(y_rot), 4, 'Y') @ rotx
    rotz = mathutils.Matrix.Rotation(math.radians(z_rot), 4, 'Z') @ roty
    return rotz


def render_and_save_dir_matrices(matrices, resolution, save_location, camera_name, img_name_format):
    where_to_save = os.path.join(PROJECT_DIRPATH, save_location)
    cam_data = {FOCAL_LENGTH: bpy.data.cameras[0].lens,
                FIELD_OF_VIEW: bpy.data.cameras[0].angle,
                FRAMES: []}
    for i, mat in enumerate(matrices):
        change_camera_loc(mat, camera_name)
        img_name = img_name_format.format(i)
        cam_data[FRAMES].append({FILENAME: img_name, TRANSFORMATION_MATRIX: np.array(mat).tolist()})
        render_and_save(os.path.join(where_to_save, img_name), resolution, resolution)
    save_json(where_to_save, cam_data)


def create_robot_images():
    # The Dataset of left to right that will be created:
    for n_pics, res in N_PICS_AND_RESOLUTION_L_TO_R:
        save_location = "RobotBlender/image_views_l_to_r/{}px_{}pics".format(res, n_pics)
        generate_l_to_r(save_location, n_pics, res, CAMERA_ROBOT, IMG_NAME_FORMAT_ROBOT,
                        VIDEO_TOTAL_X_DISTANCE_L_TO_R_ROBOT, VIDEO_Z_CLOSEST_DISTANCE_ROBOT)
        print("Done, generate_robot_l_to_r - n_pics: {}, res: {}".format(n_pics, res))

    # The Dataset of spheres that will be created:
    for n_pics, res in N_PICS_AND_RESOLUTION_SPHERE:
        save_location = "RobotBlender/image_views_sphere/{}px_{}pics".format(res, n_pics)
        x_degrees = [0, 45, -45]
        render_sphere_images(n_pics, res, save_location, x_degrees, VIDEO_SPHERE_RADIUS_ROBOT, CAMERA_ROBOT,
                             IMG_NAME_FORMAT_ROBOT)
        print("Done, generate_robot_sphere_matrices - n_pics: {}, res: {}".format(n_pics, res))

    # The Dataset of full circle that will be created:
    for n_pics, res in N_PICS_AND_RESOLUTION_FULL_CIRCLE:
        save_location = "RobotBlender/image_views_full_circle/{}px_{}pics".format(res, n_pics)
        render_full_circle(n_pics, res, save_location, VIDEO_SPHERE_RADIUS_ROBOT, CAMERA_ROBOT,
                           IMG_NAME_FORMAT_ROBOT)
        print("Done, generate_robot_full_circle - n_pics: {}, res: {}".format(n_pics, res))



#############################################################################################################
# Main for blender to run from the Blender program:
###################################################
create_robot_images()

#############################################################################################################
