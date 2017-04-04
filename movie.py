import cv2
import numpy as np

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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    undist = cc.undistort(img, mtx, dist)
    threshold_binary = th.combine(undist)

    t = pt.Transformer((threshold_binary.shape[1], threshold_binary.shape[0]))
    threshold_warp = t.warp(threshold_binary)

    finder = lf.LaneFinder(threshold_warp)
    finder.find(False)
    finder.find2(False)

    lane_layer_warp = np.zeros_like(undist)
    finder.draw_layer(lane_layer_warp)

    lane_layer = t.unwarp(lane_layer_warp)
    result = cv2.addWeighted(undist, 1, lane_layer, 0.3, 0)
    finder.draw_text(result)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result


def save_movie():
    for video_path in data.get_video_paths():
        video_clip = VideoFileClip(video_path)
        white_clip = video_clip.fl_image(pipeline)
        white_clip.write_videofile("output_" + video_path, audio=False)

save_movie()