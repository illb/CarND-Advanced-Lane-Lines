import cv2
import numpy as np

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the absolute value of the derivative or gradient
    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag) / 255
    grad_mag = (grad_mag / scale_factor).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output


def hls_select(img, h_thresh=(0, 255), l_thresh=(0, 255), s_thresh=(0, 255)):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(s)
    binary_output[(h >= h_thresh[0]) & (h <= h_thresh[1]) & (l >= l_thresh[0]) & (l <= l_thresh[1]) & (s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    return binary_output

GRADX_THRESH = (20, 120)
GRADY_THRESH = (60, 110)
MAG_THRESH = (30, 100)
MAG_KERNEL = 9
DIR_THRESH = (0.7, 1.2)
DIR_KERNEL = 11
HLS_L_THRESH1 = (200, 255)
HLS_S_THRESH1 = (130, 255)
HLS_H_THRESH2 = (10, 40)
HLS_S_THRESH2 = (120, 160)

def combine(image):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', thresh=GRADX_THRESH)
    grady = abs_sobel_thresh(image, orient='y', thresh=GRADY_THRESH)
    mag_binary = mag_thresh(image, sobel_kernel=MAG_KERNEL, thresh=MAG_THRESH)
    dir_binary = dir_threshold(image, sobel_kernel=DIR_KERNEL, thresh=DIR_THRESH)
    hls_white_binary = hls_select(image, l_thresh=HLS_L_THRESH1, s_thresh=HLS_S_THRESH1)
    hls_yellow_binary = hls_select(image, h_thresh=HLS_H_THRESH2, s_thresh=HLS_S_THRESH2)

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_white_binary == 1) | (hls_yellow_binary == 1)] = 1
    return combined


def bin2gray(img):
    result = np.zeros_like(img)
    result[(img > 0)] = 255
    return result
