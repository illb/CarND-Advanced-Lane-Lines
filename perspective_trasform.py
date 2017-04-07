import cv2
import numpy as np

class Transformer:
    def __init__(self, img_size):
        self.img_size = img_size
        self.src, self.dst = self.src_dst_rect(img_size)

    def src_dst_rect(self, img_size):
        w, h = img_size

        src_top_margin = h // 2 + 94
        src_upper_left_right_margin = 582
        src_lower_left_right_margin = 146

        dst_left_right_margin = 160

        src = np.float32([[src_upper_left_right_margin, src_top_margin], [w - src_upper_left_right_margin, src_top_margin],
                          [w - src_lower_left_right_margin, h], [src_lower_left_right_margin, h]])

        dst = np.float32([[dst_left_right_margin, 0], [w - dst_left_right_margin, 0],
                          [w - dst_left_right_margin, h], [dst_left_right_margin, h]])

        return src, dst

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
        return unwarped
