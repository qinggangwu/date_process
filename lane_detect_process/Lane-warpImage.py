

import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
# from calibration import load_calibration
from copy import copy


# left_lane = Lane()
# right_lane = Lane()
frame_width = 1920
frame_height = 1080

input_scale = 4
output_frame_scale = 4
N = 4 # buffer previous N lines

# fullsize:1280x720
x = [219, 1676, 1060, 860]
y = [1079, 1079, 452, 452]
X = [435, 1485, 1485, 435]
Y = [1079, 1079, 0, 0]


src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / input_scale)
dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) / input_scale)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)


X_b = [861, 1059, 1059, 861]
Y_b = [1079, 1079, 0, 0]
src_ = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) / (input_scale*2))
dst_ = np.floor(np.float32([[X_b[0], Y_b[0]], [X_b[1], Y_b[1]],[X_b[2], Y_b[2]], [X_b[3], Y_b[3]]]) / (input_scale*2))
M_b = cv2.getPerspectiveTransform(src_, dst_)

def warper(img, M):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

# 读取图片
image = cv2.imread('/home/wqg/data/lane_test/2022.3.22-3_16410.jpg')
image = cv2.resize(image, (0, 0), fx=1 / input_scale, fy=1 / input_scale)

binary_warped = warper(image, M)

cv2.imshow('binary_warped',binary_warped)
# cv2.waitKey(0)


newwarp = cv2.warpPerspective(binary_warped, M_inv, (int(frame_width/input_scale), int(frame_height/input_scale)))
# Combine the result with the original image    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
newwarp_ = cv2.resize(newwarp,None, fx=input_scale/output_frame_scale, fy=input_scale/output_frame_scale, interpolation = cv2.INTER_LINEAR)

cv2.imshow('newwarp_',newwarp_)
# cv2.waitKey(0)

binary_sub = np.zeros_like(binary_warped)
binary_sub[:, int(150 / input_scale):int(-80 / input_scale)] = binary_warped[:,
                                                                   int(150 / input_scale):int(-80 / input_scale)]



undist_ori = newwarp_

undist_birdview = warper(cv2.resize(undist_ori, (0, 0), fx=1 / 2, fy=1 / 2), M_b)
undist_birdview2 = warper(cv2.resize(image, (0, 0), fx=1 / 2, fy=1 / 2), M_b)
# undist_birdview = warper(undist_ori, M_b)

cv2.imshow('undist_birdview',cv2.resize(undist_birdview, (0, 0), fx= 4, fy=4))
cv2.imshow('undist_birdview2',cv2.resize(undist_birdview2, (0, 0), fx= 4, fy=4))
cv2.waitKey(0)

