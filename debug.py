import cv2
import os
import numpy as np

import data
import camera_calibration as cc
import perspective_trasform as pt
import threshold as th

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

def draw_perspective_transform_rect(fname, img):
    src, dst = pt.src_dst_rect((img.shape[1], img.shape[0]))

    src_pts = src.astype(np.int32).reshape((-1, 1, 2))
    dst_pts = dst.astype(np.int32).reshape((-1, 1, 2))

    img = cv2.polylines(img, [src_pts], True, (255, 0, 0), 4)
    img = cv2.polylines(img, [dst_pts], True, (0, 0, 255), 4)

    cv2.imwrite('./debug/perspective_transform_rect_' + os.path.basename(fname), img)

def draw_text(img, msg):
    cv2.putText(img, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_perspective_transform_warp(img):
    warp, M, Minv = pt.warp(img)
    draw_text(warp, 'wrap')
    cv2.imwrite('./debug/perspective_transform_warp_' + os.path.basename(fname), warp)


def draw_threshold(img):
    grad_binary = th.abs_sobel_thresh(img, orient='x', thresh=(20, 100))
    cv2.imwrite('./debug/threshold1_' + os.path.basename(fname), bin2gray(grad_binary))

    mag_binary = th.mag_thresh(img, sobel_kernel=3, thresh=(30, 100))
    cv2.imwrite('./debug/threshold2_' + os.path.basename(fname), bin2gray(mag_binary))

    dir_binary = th.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    cv2.imwrite('./debug/threshold3_' + os.path.basename(fname), bin2gray(dir_binary))

    combined_binary = th.combine(img)
    cv2.imwrite('./debug/threshold4_' + os.path.basename(fname), bin2gray(combined_binary))

    hls_binary = th.hls_select_s(img, thresh=(90, 255))
    cv2.imwrite('./debug/threshold5_' + os.path.basename(fname), bin2gray(hls_binary))


##################################################

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

for fname in data.get_test_paths():
    img = cv2.imread(fname)
    undist = cc.undistort_with_corners(img, objpoints, imgpoints)

    #draw_perspective_transform_rect(fname, img)

    draw_threshold(undist)

    #draw_perspective_transform_warp(undist)
    print("finished : {}".format(fname))



