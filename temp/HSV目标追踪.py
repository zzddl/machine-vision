#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/27
# @file HSV目标追踪.py

import cv2
import numpy as np

img = cv2.imread('circle.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([11, 43, 46])
upper = np.array([25, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('hsv', res)
cv2.waitKey(0)
cv2.destroyAllWindows()