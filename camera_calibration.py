import numpy as np
import cv2
import os
import data

_GRID_X_NUM = 9
_GRID_Y_NUM = 6

def find_corners():
    # ref : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((_GRID_Y_NUM * _GRID_X_NUM, 3), np.float32)
    objp[:, :2] = np.mgrid[0:_GRID_X_NUM, 0:_GRID_Y_NUM].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in data.get_calibration_paths():
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print("started finding chessboard : {}".format(fname))
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (_GRID_X_NUM, _GRID_Y_NUM), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        print("finished finding chessboard : {}, success={}".format(fname, ret))

    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, img_size):
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


# performs the camera calibration, image distortion correction and
# returns the undistorted image
def undistort_with_corners(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    mtx, dist = calibrate_camera(objpoints, imgpoints, img_size)
    return undistort(img, mtx, dist)

import pickle

_CORNERS_PICKLE_PATH = "camera_calibration_corners.p"

def save_corners(objpoints, imgpoints):
    d = { "objpoints": objpoints, "imgpoints": imgpoints }
    with open(_CORNERS_PICKLE_PATH, "wb") as f:
        pickle.dump(d, f)

def load_corners():
    # Read in the saved objpoints and imgpoints
    d = {}
    with open(_CORNERS_PICKLE_PATH, "rb") as f:
        d = pickle.load(f)
    objpoints = d["objpoints"]
    imgpoints = d["imgpoints"]
    return objpoints, imgpoints
