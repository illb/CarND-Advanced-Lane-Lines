import cv2
import os
import numpy as np

import data
import camera_calibration as cc
import perspective_trasform as pt
import threshold as th
import lane_find as lf

img_size = (1280, 720)
output_dir = "./debug"

objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)

test_paths = ["./test_images/hard_test1.jpg"]


def mask(undist, img):
    res = undist.copy()
    res[(img > 0)] = 255
    return res

def mask_r(undist, img):
    res = undist.copy()
    res[(img == 0)] = 255
    return res

def save_threshold():

    for fname in test_paths:
        img = cv2.imread(fname)
        undist = cc.undistort(img, mtx, dist)

        cv2.imwrite(output_dir + '/undist_' + os.path.basename(fname), undist)

        for s in range(0, 150, 10):
            gradx_binary = th.abs_sobel_thresh(undist, orient='x', thresh=(s, s+100))
            cv2.imwrite(output_dir + "/threshold_gradx_{}_".format(s) + os.path.basename(fname), mask(undist, gradx_binary))

        for s in range(0, 200, 10):
            grady_binary = th.abs_sobel_thresh(undist, orient='y', thresh=(s, s+50))
            cv2.imwrite(output_dir + "/threshold_grady_{}_".format(s) + os.path.basename(fname), mask(undist, grady_binary))

        for s in range(0, 180, 10):
            mag_binary = th.mag_thresh(undist, sobel_kernel=13, thresh=(s, s+70))
            cv2.imwrite(output_dir + "/threshold_mag_{}_".format(s) + os.path.basename(fname), mask(undist, mag_binary))

        for s in np.arange(0.0, 2.0, 0.1):
            dir_binary = th.dir_threshold(undist, sobel_kernel=17, thresh=(s, s + 0.6))
            cv2.imwrite(output_dir + "/threshold_dir_{:.2f}_".format(s) + os.path.basename(fname), mask(undist, dir_binary))

        for s in range(0, 200, 10):
            hls_binary = th.hls_select(undist, s_thresh=(s, s + 55))
            cv2.imwrite(output_dir + "/threshold_hls_hs1_{}_".format(s) + os.path.basename(fname), mask(undist, hls_binary))

        for s in range(0, 100, 10):
            hls_binary = th.hls_select(undist, s_thresh=(s, s + 60))
            cv2.imwrite(output_dir + "/threshold_hls_hs2_{}_".format(s) + os.path.basename(fname), mask_r(undist, hls_binary))


        combined_binary = th.combine(undist)
        cv2.imwrite(output_dir + '/threshold_combined_' + os.path.basename(fname), mask(undist, combined_binary))

save_threshold()