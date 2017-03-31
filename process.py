import camera_calibration as cc

######################################
# 01) Camera Calibration

objpoints, imgpoints = cc.find_corners()
cc.save_corners(objpoints, imgpoints)

######################################
# 02) Distrotion Correction



######################################
# 03) Color & Gradient threshold



######################################
# 04) Perspective Transform


import cv2
import numpy as np
import os
import data


def undist(objpoints, imgpoints):
    fnames = []

    for fname in data.get_test_paths():
        img = cv2.imread(fname)
        dst = cc.undistort_with_corners(img, objpoints, imgpoints)
        outname = './debug/undistorted_' + os.path.basename(fname)
        cv2.imwrite(outname, dst)
        print("undistort_calibration : {}".format(outname))


objpoints, imgpoints = cc.load_corners()
undist(objpoints, imgpoints)

# img = mpimg.imread('./test_images/test1.jpg')
# undistorted = cc.undistort_with_corners(img, objpoints, imgpoints)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(undistorted)
# ax2.set_title('Undistorted Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)