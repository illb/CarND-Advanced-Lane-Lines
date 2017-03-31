import cv2
import os
import numpy as np

import data
import camera_calibration as cc
import perspective_trasform as pt
import threshold as th
import lane_find as lf

# Draw and display the corners
def draw_chessboard_corners():
    for fname in data.get_calibration_paths():
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cc._GRID_X_NUM, cc._GRID_Y_NUM), None)
        if ret:
            cv2.drawChessboardCorners(img, (cc._GRID_X_NUM, cc._GRID_Y_NUM), corners, ret)
            cv2.imwrite('./debug/corners_' + os.path.basename(fname), img)


def draw_perspective_transform_rect(fname, t, img):
    src, dst = t.src, t.dst

    src_pts = src.astype(np.int32).reshape((-1, 1, 2))
    dst_pts = dst.astype(np.int32).reshape((-1, 1, 2))

    img = cv2.polylines(img, [src_pts], True, (255, 0, 0), 4)
    img = cv2.polylines(img, [dst_pts], True, (0, 0, 255), 4)

    cv2.imwrite('./debug/perspective_transform_rect_' + os.path.basename(fname), img)


def draw_text(img, msg):
    cv2.putText(img, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def draw_perspective_transform_warp(fname, t, img):
    warp = t.warp(img)
    cv2.imwrite('./debug/perspective_transform_warp_' + os.path.basename(fname), warp)


def combine(image):
    # Apply each of the thresholding functions
    gradx = th.abs_sobel_thresh(image, orient='x', thresh=(20, 100))
    grady = th.abs_sobel_thresh(image, orient='y', thresh=(5, 30))
    mag_binary = th.mag_thresh(image, sobel_kernel=3, thresh=(50, 100))
    dir_binary = th.dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.2))
    hls_binary = th.hls_select_s(image, thresh=(170, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    return combined


def draw_threshold(fname, img):
    gradx_binary = th.abs_sobel_thresh(img, orient='x', thresh=(20, 100))
    cv2.imwrite('./debug/threshold_gx_' + os.path.basename(fname), bin2gray(gradx_binary))

    grady_binary = th.abs_sobel_thresh(img, orient='y', thresh=(5, 30))
    cv2.imwrite('./debug/threshold_gy_' + os.path.basename(fname), bin2gray(grady_binary))

    gradxy_binary = np.zeros_like(gradx_binary)
    gradxy_binary[(gradx_binary == 1) & (grady_binary == 1)] = 1
    cv2.imwrite('./debug/threshold_gxy_' + os.path.basename(fname), bin2gray(gradxy_binary))

    mag_binary = th.mag_thresh(img, sobel_kernel=3, thresh=(50, 100))
    cv2.imwrite('./debug/threshold_m_' + os.path.basename(fname), bin2gray(mag_binary))

    dir_binary = th.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.2))
    cv2.imwrite('./debug/threshold_d_' + os.path.basename(fname), bin2gray(dir_binary))

    md_binary = np.zeros_like(mag_binary)
    md_binary[(mag_binary == 1) & (dir_binary == 1)] = 1
    cv2.imwrite('./debug/threshold_md_' + os.path.basename(fname), bin2gray(md_binary))

    hls_binary = th.hls_select_s(img, thresh=(170, 255))
    cv2.imwrite('./debug/threshold_s_' + os.path.basename(fname), bin2gray(hls_binary))

    combined_binary = combine(img)
    cv2.imwrite('./debug/threshold_c_' + os.path.basename(fname), bin2gray(combined_binary))

# objpoints, imgpoints = cc.find_corners()
# cc.save_corners(objpoints, imgpoints)
# print(objpoints, imgpoints)

objpoints, imgpoints = cc.load_corners()
#print(objpoints, imgpoints)

# draw_chessboard_corners()

def bin2gray(img):
    img2 = np.zeros_like(img)
    img2[(img > 0)] = 255
    return img2

def debug_test():
    for fname in data.get_test_paths():
        img = cv2.imread(fname)
        undist = cc.undistort_with_corners(img, objpoints, imgpoints)

        threshold_binary = combine(undist)
        draw_threshold(fname, undist)

        t = pt.Transformer((threshold_binary.shape[1], threshold_binary.shape[0]))
        threshold_gray = bin2gray(threshold_binary)
        threshold_warp = t.warp(threshold_gray)
        # draw_perspective_transform_rect(fname, threshold_gray)
        draw_perspective_transform_warp(fname, t, threshold_gray)

        finder = lf.LaneFinder(threshold_warp)
        found1 = finder.find()
        cv2.imwrite('./debug/result1_' + os.path.basename(fname), found1)
        found2 = finder.find2()
        cv2.imwrite('./debug/result2_' + os.path.basename(fname), found2)

        lane_layer_warp = np.zeros_like(found1)
        finder.draw_layer(lane_layer_warp)

        found3 = cv2.addWeighted(found1, 1, lane_layer_warp, 0.5, 0)

        cv2.imwrite('./debug/result3_' + os.path.basename(fname), found3)

        lane_layer = t.unwarp(lane_layer_warp)
        result4 = cv2.addWeighted(np.float64(undist), 1, lane_layer, 0.5, 0)
        finder.draw_text(result4)
        cv2.imwrite('./debug/result4_' + os.path.basename(fname), result4)

        print("finished : {}".format(fname))

from moviepy.editor import VideoFileClip

def pipeline(img):
    undist = cc.undistort_with_corners(img, objpoints, imgpoints)
    threshold_binary = combine(undist)

    t = pt.Transformer((threshold_binary.shape[1], threshold_binary.shape[0]))
    threshold_gray = bin2gray(threshold_binary)
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


def movie():
    video_path = data.get_video_paths()[0]
    video_clip = VideoFileClip(video_path)
    white_clip = video_clip.fl_image(pipeline)
    white_clip.write_videofile("output_" + video_path, audio=False)

movie()