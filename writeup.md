##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

###Source Files
* debug.py : sandbox for debugging
* process.py : full pipeline process
* movie.py : make the output video
* modules
  * data.py : data list functions
  * camera_calibration.py : camera calibration functions
  * threshold.py : threshold filter functions
  * perspective_transform.py : perspective transform class
  * lane_find.py : lane find class

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

* camera_calibration.py 
  * `find_corners` function (lines 9 ~ 34) can find objpoints, imgpoints
    * objpoints : (x, y, z=0) coordinates of the chessboard corners in the world
    * imgpoints : found corners
  * `calibrate_camera` function (line 37 ~ 40) can compute the camera calibration (mtx : 3x3 floating-point camera matrix, dist : vector of distortion coefficients) with objpoints, imgpoints
  * `undistort` function can correct image distortion with mtx, dist
* original image

![calibration1](./camera_cal/calibration3.jpg =300x)

* undistorted image

![calibration_undist1](./output_images/undistorted_calibration3.jpg =300x)

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![pipeline_undist](./output_images/undistorted_test4.jpg =300x)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

* color combination of color and gradient thresholds
  * there are corresponding functions in `threshold.py` 
  * the pipeline proceeds in the following order
  
* sobel x, y
  * `abs_sobel_thresh` function (lines 6 ~ 21) 

![threshold_gradx](./output_images/threshold_gradx_test4.jpg =300x)
![threshold_grady](./output_images/threshold_grady_test4.jpg =300x)

* sobel x and y
  * `(gradx == 1) & (grady == 1)` (line 91)

![threshold_gradxy](./output_images/threshold_gradxy_test4.jpg =300x)

* magnitude of the gradient
  * `mag_thresh` function (lines 25 ~ 39)
  
![threshold_mag](./output_images/threshold_mag_test4.jpg =300x)

* direction of the gradient
  * `dir_threshold` function (lines 45 ~ 57)

![threshold_dir](./output_images/threshold_dir_test4.jpg =300x)

* magnitude of the gradient and direction of the gradient
  * `(mag_binary == 1) & (dir_binary == 1)` (line 90)

![threshold_magdir](./output_images/threshold_magdir_test4.jpg =300x)

* thresholds the S-channel of HLS
  * `hls_select_s` function (lines 62 ~ 70)

![threshold_hls_s](./output_images/threshold_hls_s_test4.jpg =300x)

![threshold_hls_s](./output_images/threshold_hls_s2_test4.jpg =300x)

* combined result
  * `((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary1 == 1) | (hls_binary2 == 1)` (line 91)

![threshold_combined](./output_images/threshold_combined_test4.jpg =300x)


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 26 through 31 in the file `perspective_transform.py`. The `warp()` function needs as inputs an image (`img`) and needs member variables source (`src`) and destination (`dst`) points. I chose the hardcode the source and destination points in the following manner:

```
w, h = img_size

src_top_margin = h // 2 + 94
src_upper_left_right_margin = 570
src_lower_left_right_margin = 146

dst_left_right_margin = 160

src = np.float32([[src_upper_left_right_margin, src_top_margin], [w - src_upper_left_right_margin, src_top_margin],
                  [w - src_lower_left_right_margin, h], [src_lower_left_right_margin, h]])

dst = np.float32([[dst_left_right_margin, 0], [w - dst_left_right_margin, 0],
                  [w - dst_left_right_margin, h], [dst_left_right_margin, h]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 454      | 160, 0        | 
| 710, 454      | 1120, 0       |
| 1134, 720     | 1120, 720     |
| 146, 720      | 160, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

* src : blue rectangle
* dst : red rectangle 

![threshold_combined](./output_images/perspective_transform_rect_test4.jpg =300x)

* warped image

![threshold_combined](./output_images/perspective_transform_warp_test4.jpg =300x)

* threshold filtered and warped image

![threshold_combined](./output_images/perspective_transform_threshold_warp_test4.jpg =300x)

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

* Find the peak of the left and right halves of the histogram. these will be the starting point for the left and right lines
  * `find_base` function (`lane_find.py` lines 23 ~ 78)

* If there are starting points found in the previous screen, weights the starting points.
The weights are in the form of normal distributions.

```
left_norm = np.int64(self._norm_histogram(x, self.leftx_base))
right_norm = np.int64(self._norm_histogram(x, self.rightx_base))
new_histogram = histogram * (left_norm + right_norm)

```

* Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

* f(y) = Ay^2 + By + C
  * `find_step1` function (`lane_find.py` lines 147 ~ 148)

![result1](./output_images/result1_test4.jpg =300x)

* Find the line again for the area you just gave the 2nd order polynomial line.
  * `find_step2` function (`lane_find.py` lines 177 ~ 259)


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 241 ~ 257 in my code in `lane_find.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 131 through 138 in my code in `process.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![result4](./output_images/result4_test4.jpg =300x)

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* hard to find lanes scenes
  * gray road
  
![hard_test](./test_images/hard_test1.jpg =300x)  

  * shadowed road
  
![hard_test](./test_images/hard_test2.jpg =300x)  

  * 2 colors of road

![hard_test](./test_images/hard_test4.jpg =300x)  
  
  * partly covered road

![hard_test](./test_images/hard_test6.jpg =300x)  

  * too bright road

![hard_test](./test_images/hard_test7.jpg =300x)  

* a few ways improvement:
  * more than two polynomials are needed for more than two bends
  * use deep learning for object (car / bike) detection
  
