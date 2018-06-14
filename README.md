# **Advanced Lane Finding**

[//]: # (Image References)

[image0]: ./examples/cover.png
[image1]: ./examples/find_corners.png
[image2]: ./examples/undistort_chessboard.png
[image3]: ./examples/undistort_raw.png
[image4]: ./examples/threshold.png
[image5]: ./examples/warped_image.png
[image6]: ./examples/fit_lines.png
[image7]: ./examples/warp_back.png

This is a brief writeup report of Self-Driving Car Engineer P4.

<img src="./examples/cover.png" width="50%" height="50%" />

---


**Steps Of This Project**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

<Br/>

## Camera Calibration

The code for this step is contained in the 2nd to 5th code cells of the IPython notebook located in `P4.ipynb`, on basis of chessboard photos. Here are some key processes:

#### 1. Find Chessboard Corners

* Prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
* Assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.
* Replicate the array of coordinates to `objp`.
* Append `objpoints` with a copy of it every time all chessboard corners in a test image are successfully detected.
* Append `imgpoints` with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 
 
![alt text][image1]

#### 2. Compute the camera calibration

* Use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
* Apply this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

<Br/>

## Pipeline (single images)

#### 1. Apply the distortion correction to raw images.

All the test images in '/test_images/' applied the distortion correction are like this:

![alt text][image3]

#### 2. Create a thresholded binary image

A combination of b channel(Lab) and L channel(HLS) thresholds generates a binary image (thresholding steps in the 7th code cell of `P4.ipynb`). 

* b channel of Lab: detect yellow line at a high channel value (about 175~255)
* L channel of HLS: detect white line by high luminance

Here's an output example for this step.

![alt text][image4]

#### 3. Apply a perspective transform ("birds-eye view")

The code for perspective transform includes a function called `transformImg()`, which appears in the 8th code cell of `P4.ipynb`. 

The `transformImg()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. Then it use `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` functions to warp images.

The hardcode of the source and destination points is in the following manner:

```python
src = np.float32(
    [[img_size[0] / 2 - 90, img_size[1] / 2 + 100],
     [img_size[0] / 2 + 90, img_size[1] / 2 + 100],
     [img_size[0] * 31 / 32 + 10, img_size[1] - 1],
     [img_size[0] / 32 - 10, img_size[1] - 1]])
dst = np.float32(
    [[img_size[0] / 8 - 60, 1],
     [img_size[0] * 7 / 8 + 60, 1],
     [img_size[0] * 7 / 8 + 60, img_size[1] - 1],
     [img_size[0] / 8 - 60, img_size[1] - 1]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 460      | 100, 1        | 
| 730, 460      | 1180, 1      |
| 1250, 719     | 1180, 719      |
| 30, 719      | 100, 719        |

Perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Detect lane pixels and fit to find the lane boundary

The code for lane pixels detect includes two functions called `binary_fit()` and `lane_boundary()`, those appear in the 11th-12th code cells of `P4.ipynb`.

* `binary_fit()` takes the warped binary image as input (`binary_warped`) and implement sliding windows method for fitting a polynomial.
* `lane_boundary()` not only takes the warped binary image as input but also takes `left_fit` and `right_fit`, which are output from `binary_fit()`. Then use `cv2.fillPoly` to fit the lane boundary.

The 2nd order polynomial kinda like this: f(y)=Ay<sup>2</sup>+By+C

![alt text][image6]

#### 5. Determine the curvature of the lane and vehicle position with respect to center

The code for curvature measurement includes a function called `measure_Curvature()`, which appear s in the 14th code cell of `P4.ipynb`.

The measurement result of the test image is:

```
Radius of Curvature = 894(m)
Vehicle is 0.54m left of center
```

#### 6. Warp the detected lane boundaries back onto the original image

The code for warpping back includes a function called `draw_Lane()`, which appears in the 16th code cell of `P4.ipynb`.

Here is an example result on the test image:

![alt text][image7]

---

## Pipeline (video)

#### Output visual display of the lane boundaries and numerical estimation

The code of video process pipeline appears in the 19th code cell of `P4.ipynb`, which includes almost all functions defined in the project:

* `undistortImg()`
* `binary_thresh()`
* `transformImg()`
* `binary_fit()`
* `lane_boundary()`
* `measure_Curvature()`
* `draw_Lane()`
* `cv2.putText()`

And the pipeline performs well on the entire project video.

Here's a [link to the project video result](./Advanced_Lane_Lines.mp4)

---

## Discussion

#### Threshold method choosing and combination are the most important key of the pipeline

The combination of S channel in HLS and the S gradients on x axis is not robust enough in this project, it would fail on some frames with shadows or  pavement color changes.

Conversely, the combination of L channel in HLS and the b channel in Lab can detect yellow and white lane lines well, so this threshold method did well in the  'project_video.mp4'.

BTW, color transforms have its shortcomings when those cases such as Strong reflection of sunshine on the front windshield, messy shadows like leaves or pavement color changes which along the lane lines.

So, finding a robust enough method to combine color transforms, gradients, or some deep neural network maybe, is much needed for the self-driving car's vision.
