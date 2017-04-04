import cv2
import os
import numpy as np

import data
import camera_calibration as cc
import perspective_trasform as pt
import threshold as th
import lane_find as lf

img_size = (1280, 720)
output_dir = "./output_images"

##################################################
# 01) Camera Calibration

# Draw and display the corners
def save_chessboard_corners():
    for fname in data.get_calibration_paths():
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cc._GRID_X_NUM, cc._GRID_Y_NUM), None)
        if ret:
            cv2.drawChessboardCorners(img, (cc._GRID_X_NUM, cc._GRID_Y_NUM), corners, ret)
            cv2.imwrite(output_dir + '/corners_' + os.path.basename(fname), img)


print("========= 01) Camera Calibration")
# objpoints, imgpoints = cc.find_corners()
# cc.save_corners(objpoints, imgpoints)

# save_chessboard_corners()

######################################
# 02) Distrotion Correction

def save_undistorted(mtx, dist):
    paths = np.append(data.get_calibration_paths(), data.get_test_paths())
    for fname in paths:
        img = cv2.imread(fname)
        dst = cc.undistort(img, mtx, dist)
        outname = output_dir + '/undistorted_' + os.path.basename(fname)
        cv2.imwrite(outname, dst)
        print("undistorted : {}".format(outname))


print("========= 02) Distrotion Correction")
objpoints, imgpoints = cc.load_corners()
mtx, dist = cc.calibrate_camera(objpoints, imgpoints, img_size)
# save_undistorted(mtx, dist)

######################################
# 03) Color & Gradient threshold

def save_threshold():
    for fname in data.get_test_paths():
        img = cv2.imread(fname)
        undist = cc.undistort(img, mtx, dist)

        gradx_binary = th.abs_sobel_thresh(undist, orient='x', thresh=th.GRADX_THRESH)
        cv2.imwrite(output_dir + '/threshold_gradx_' + os.path.basename(fname), th.bin2gray(gradx_binary))

        grady_binary = th.abs_sobel_thresh(undist, orient='y', thresh=th.GRADY_THRESH)
        cv2.imwrite(output_dir + '/threshold_grady_' + os.path.basename(fname), th.bin2gray(grady_binary))

        gradxy_binary = np.zeros_like(gradx_binary)
        gradxy_binary[(gradx_binary == 1) & (grady_binary == 1)] = 1
        cv2.imwrite(output_dir + '/threshold_gradxy_' + os.path.basename(fname), th.bin2gray(gradxy_binary))

        mag_binary = th.mag_thresh(undist, sobel_kernel=th.MAG_KERNEL, thresh=th.MAG_THRESH)
        cv2.imwrite(output_dir + '/threshold_mag_' + os.path.basename(fname), th.bin2gray(mag_binary))

        dir_binary = th.dir_threshold(undist, sobel_kernel=th.DIR_KERNEL, thresh=th.DIR_THRESH)
        cv2.imwrite(output_dir + '/threshold_dir_' + os.path.basename(fname), th.bin2gray(dir_binary))

        md_binary = np.zeros_like(mag_binary)
        md_binary[(mag_binary == 1) & (dir_binary == 1)] = 1
        cv2.imwrite(output_dir + '/threshold_magdir_' + os.path.basename(fname), th.bin2gray(md_binary))

        hls_binary = th.hls_select_s(undist, thresh=th.HLS_S_THRESH)
        cv2.imwrite(output_dir + '/threshold_hls_s_' + os.path.basename(fname), th.bin2gray(hls_binary))

        combined_binary = th.combine(undist)
        cv2.imwrite(output_dir + '/threshold_combined_' + os.path.basename(fname), th.bin2gray(combined_binary))

print("========= 03) Color & Gradient threshold")
save_threshold()

######################################
# 04) Perspective Transform

def draw_perspective_transform_rect(t, img):
    img2 = img.copy()
    src, dst = t.src, t.dst

    src_pts = src.astype(np.int32).reshape((-1, 1, 2))
    dst_pts = dst.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(img2, [src_pts], True, (255, 0, 0), 4)
    cv2.polylines(img2, [dst_pts], True, (0, 0, 255), 4)
    return img2


def save_perspective_transform():
    for fname in data.get_test_paths():
        img = cv2.imread(fname)
        undist = cc.undistort(img, mtx, dist)

        t = pt.Transformer((undist.shape[1], undist.shape[0]))
        perspective_rect = draw_perspective_transform_rect(t, undist)
        cv2.imwrite(output_dir + '/perspective_transform_rect_' + os.path.basename(fname), perspective_rect)

        warp = t.warp(undist)
        cv2.imwrite(output_dir + '/perspective_transform_warp_' + os.path.basename(fname), warp)

        threshold_binary = th.combine(undist)
        threshold_warp = t.warp(threshold_binary)

        cv2.imwrite(output_dir + '/perspective_transform_threshold_warp_' + os.path.basename(fname), th.bin2gray(threshold_warp))

print("========= 04) Perspective Transform")
save_perspective_transform()

######################################
# 05) Find Lanes

def map_lane(t, finder, img):
    lane_layer_warp = np.zeros_like(img)
    finder.draw_layer(lane_layer_warp)

    lane_layer = t.unwarp(lane_layer_warp)
    result = cv2.addWeighted(img, 1, lane_layer, 0.3, 0)
    finder.draw_text(result)
    return result


def save_found_lanes():
    for fname in data.get_test_paths():
        img = cv2.imread(fname)
        undist = cc.undistort(img, mtx, dist)
        threshold_binary = th.combine(undist)

        t = pt.Transformer((threshold_binary.shape[1], threshold_binary.shape[0]))
        threshold_warp = t.warp(threshold_binary)

        finder = lf.LaneFinder(threshold_warp)
        result1 = finder.find()
        cv2.imwrite(output_dir + '/result1_' + os.path.basename(fname), result1)
        result2 = finder.find2()
        cv2.imwrite(output_dir + '/result2_' + os.path.basename(fname), result2)

        lane_layer = np.zeros_like(result1)
        finder.draw_layer(lane_layer)
        result3 = cv2.addWeighted(result1, 1, lane_layer, 0.3, 0)
        cv2.imwrite(output_dir + '/result3_' + os.path.basename(fname), result3)

        result4 = map_lane(t, finder, undist)
        cv2.imwrite(output_dir + '/result4_' + os.path.basename(fname), result4)

print("========= 05) Find Lanes")
save_found_lanes()