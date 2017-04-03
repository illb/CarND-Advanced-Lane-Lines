import cv2
import numpy as np
import os

import data
import camera_calibration as cc
import perspective_trasform as pt
import threshold as th
import lane_find as lf

img_size = (1280, 720)
output_dir = "./output_images"
from moviepy.editor import VideoFileClip

######################################
# Distrotion Correction
objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)

def pipeline(img):
    undist = cc.undistort(img, mtx, dist)
    threshold_binary = th.combine(undist)

    t = pt.Transformer((threshold_binary.shape[1], threshold_binary.shape[0]))
    threshold_gray = th.bin2gray(threshold_binary)
    threshold_warp = t.warp(threshold_gray)

    finder = lf.LaneFinder(threshold_warp)
    finder.find(False)
    finder.find2(False)

    undist_float = np.float64(undist)

    lane_layer_warp = np.zeros_like(undist_float)
    finder.draw_layer(lane_layer_warp)

    lane_layer = t.unwarp(lane_layer_warp)
    result = cv2.addWeighted(undist_float, 1, lane_layer, 0.5, 0)
    finder.draw_text(result)
    return result


def save_movie():
    video_path = data.get_video_paths()[0]
    video_clip = VideoFileClip(video_path)
    white_clip = video_clip.fl_image(pipeline)
    white_clip.write_videofile("output_" + video_path, audio=False)

save_movie()