import numpy as np
import cv2
import threshold as th

class LaneFinder:
    def __init__(self, binary_warped):
        self.binary_warped = binary_warped

        self.has_last = False
        self.leftx_base = None
        self.lefty_base = None

    def _norm_histogram(self, x, center):
        from scipy.stats import norm
        sigma = 40.0
        dist = norm(center, sigma)
        res = dist.pdf(x) * 20.0
        return res.reshape(x.shape[0])

    def find_base(self, debug=True):
        binary_warped = self.binary_warped
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.int64(np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)

        w, h = binary_warped.shape[1], binary_warped.shape[0]
        x = np.arange(0, w).reshape(w, 1)
        if self.has_last:
            n1 = np.int64(self._norm_histogram(x, self.leftx_base))
            n2 = np.int64(self._norm_histogram(x, self.rightx_base))
            histogram = histogram * (n1 + n2)

        out_img = None
        if debug:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

            # draw the histogram of bottom half with red
            y = h - histogram.reshape(w, 1)
            pts = np.concatenate((x, y), axis=1)
            cv2.polylines(out_img, np.int32([pts]), False, (0, 0, 255), 6)

            if self.has_last:
                y = (h - n1.reshape(w, 1))
                pts = np.concatenate((x, y), axis=1)
                cv2.polylines(out_img, np.int32([pts]), False, (255, 0, 0), 6)

                y = (h - n2.reshape(w, 1))
                pts = np.concatenate((x, y), axis=1)
                cv2.polylines(out_img, np.int32([pts]), False, (255, 0, 0), 6)

                sum = histogram * (n1 + n2)
                y = (h - sum).reshape(w, 1)
                pts = np.concatenate((x, y), axis=1)
                cv2.polylines(out_img, np.int32([pts]), False, (0, 255, 0), 6)

        self.has_last = True
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return out_img


    def find(self, debug=True):
        self.find_base(False)

        # Assuming you have created a warped binary image called "binary_warped"
        binary_warped = self.binary_warped

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        out_img = None
        if debug:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            if debug:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if debug:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            y = ploty.reshape(ploty.shape[0], 1)
            lx = left_fitx.reshape(left_fitx.shape[0], 1)
            rx = right_fitx.reshape(right_fitx.shape[0], 1)
            left_pts = np.concatenate((lx, y), axis=1)
            right_pts = np.concatenate((rx, y), axis=1)

            cv2.polylines(out_img, np.int32([left_pts]), False, (0, 255, 255), 8)
            cv2.polylines(out_img, np.int32([right_pts]), False, (0, 255, 255), 8)

        self.left_fit = left_fit
        self.right_fit = right_fit

        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty

        return out_img

    def find2(self, debug=True):
        binary_warped = self.binary_warped
        left_fit = self.left_fit
        right_fit = self.right_fit
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 90
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img = None
        window_img = None
        result = None
        if debug:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if debug:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        self.left_fit = left_fit
        self.right_fit = left_fit

        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        self.left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        self.right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        self.center_diff = (camera_center - binary_warped.shape[1] / 2) * xm_per_pix

        return result

    def draw_layer(self, img):
        y = self.ploty.reshape(self.ploty.shape[0], 1)
        lx = self.left_fitx.reshape(self.left_fitx.shape[0], 1)
        rx = self.right_fitx.reshape(self.right_fitx.shape[0], 1)
        left_pts = np.concatenate((lx, y), axis=1)
        right_pts = np.flip(np.concatenate((rx, y), axis=1), axis=0)
        pts = np.append(left_pts, right_pts, axis=0)

        cv2.fillPoly(img, np.int32([pts]), (0, 255, 255))
        cv2.polylines(img, np.int32([left_pts]), False, (255, 0, 0), 8)
        cv2.polylines(img, np.int32([right_pts]), False, (0, 0, 255), 8)


    def draw_text(self, img):
        # display the info
        cv2.putText(img, 'Radius of Curvature = ' + str(round((self.left_curverad + self.right_curverad) / 2, 3)) + '(m)',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Vehicle is ' + str(round(self.center_diff, 3)) + '(m) off center', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
