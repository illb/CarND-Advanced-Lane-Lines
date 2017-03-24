import cv2
import os
import numpy as np

import data
import camera_calibration as cc
import perspective_trasform as pt

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


##################################################

# objpoints, imgpoints = cc.find_corners()
# cc.save_corners(objpoints, imgpoints)
# print(objpoints, imgpoints)

objpoints, imgpoints = cc.load_corners()
#print(objpoints, imgpoints)

# draw_chessboard_corners()



for fname in data.get_test_paths():
    img = cv2.imread(fname)
    undist = cc.undistort_with_corners(img, objpoints, imgpoints)
    draw_perspective_transform_rect(fname, img)
    warp, M, Minv = pt.warp(undist)
    draw_text(warp, 'wrap')
    cv2.imwrite('./debug/perspective_transform_warp_' + os.path.basename(fname), warp)



