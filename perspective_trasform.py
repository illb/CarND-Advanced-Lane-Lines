import cv2
import numpy as np

def src_dst_rect(img_size):
    w, h = img_size

    src_top_margin = h // 2 + 80
    src_upper_left_right_margin = 540
    src_lower_left_right_margin = 40

    dst_left_right_margin = 40

    src = np.float32([[src_upper_left_right_margin, src_top_margin], [w - src_upper_left_right_margin, src_top_margin],
                      [w - src_lower_left_right_margin, h], [src_lower_left_right_margin, h]])

    dst = np.float32([[dst_left_right_margin, 0], [w - dst_left_right_margin, 0],
                      [w - dst_left_right_margin, h], [dst_left_right_margin, h]])

    return src, dst

def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src, dst = src_dst_rect(img_size)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


