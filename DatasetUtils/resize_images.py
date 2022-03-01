"""
This is a helper script that resizes the images found in the specified directory in Assets, and name the images by the
specified image name.
"""

import os
from pathlib import Path
import cv2
from imageio import imread


def is_image_filename(filename):
    return filename.endswith('.jpg') or filename.endswith('.png')


def save_images_with_new_resolution(image_filenames, new_resolution, path_to_images, where_to_save, image_name):
    for i, image_filename in enumerate(image_filenames):
        image = imread(path_to_images / image_filename)
        new_res_image = cv2.resize(image, dsize=(new_resolution, new_resolution), interpolation=cv2.INTER_AREA)
        new_res_image = cv2.cvtColor(new_res_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(where_to_save / f"{image_name}_{i:03}.jpg"), new_res_image)


def main():
    """
    Change the dir_with_images directory name in Assets and the new_images_name to your liking.
    """
    ###################################
    # Uses this 2 following parameters:
    dir_with_images = 'dir_name'    # TODO: put path to directory.
    new_images_name = "name"         # TODO: put video name
    path_to_assets = Path('Assets')
    ###################################

    os.chdir('..')
    path_to_images = path_to_assets / dir_with_images
    image_filenames = [filename for filename in os.listdir(path_to_images) if is_image_filename(filename)]

    resolutions = [50, 100, 256, 512]
    for new_resolution in resolutions:
        where_to_save = path_to_assets / dir_with_images / '{}px_{}pics'.format(new_resolution, len(image_filenames))
        if not os.path.exists(where_to_save):
            os.makedirs(where_to_save)
        save_images_with_new_resolution(image_filenames, new_resolution, path_to_images, where_to_save, new_images_name)
    print("Done!")


if __name__ == '__main__':
    main()
