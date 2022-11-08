#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/11
# @file 大津二值化算法.py

import cv2
import numpy as np

# 读取图像
img = cv2.imread('lenna.png').astype(np.float)

# 灰度化图像
out  = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
out = out.astype(np.uint8)

# OSTU法二值化
h, w = out.shape # 获取行列值
max_sigma = 0
max_t = 0

for t in range(255):

    # 按照阈值将像素值分为class0和class1
    c0 = img[np.where(out < t)]
    c1 = img[np.where(out >= t)]

    # 求均值
    m0 = c0.mean() if len(c0) > 0 else 0
    m1 = c1.mean() if len(c1) > 0 else 0

    # 两个class分别所占的比例
    w0 = len(c0) / (h * w)
    w1 = len(c1) / (h * w)

    # 求类间方差
    sigma = w0 * w1 * ((m0 - m1) ** 2)

    if sigma > max_sigma:
        max_sigma = sigma
        max_t = t

print(max_sigma)
print(max_t)

# 图像二值化
out[out < max_t] = 0
out[out >= max_t] = 255
cv2.imshow('OSTU', out)
cv2.waitKey(0)
