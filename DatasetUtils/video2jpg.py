"""
This script split a video to frames in the same place.
"""

import os
from pathlib import Path

import cv2


def main():
    #####################################################
    # Change directory names here for appropriate folders.
    video_dir = Path('Assets') / 'directory'    # TODO: put path to directory.
    video_name = Path('video.mp4')              # TODO: put video name
    save_1_frame_out_of_n = 5       # Interval size - used to reduce amount of frames to save.
    #####################################################
    os.chdir('..')

    video_full_path = str(video_dir / video_name)
    video_cap = cv2.VideoCapture(video_full_path)

    assert os.path.exists(video_full_path), "Cannot find " + video_full_path

    success, image = video_cap.read()
    if not success:
        print("Error, cannot read video {}".format(video_full_path))

    count = 0
    while success:
        if count % save_1_frame_out_of_n == 0:
            frame_name = "{}_{:05d}.jpg".format(video_name.stem, count // save_1_frame_out_of_n)
            fullpath_frame = str(video_dir / frame_name)
            cv2.imwrite(fullpath_frame, image)  # save frame as JPEG file
        success, image = video_cap.read()
        count += 1
    print("Done writing", count // save_1_frame_out_of_n, "frames in", video_dir)


if __name__ == '__main__':
    main()
