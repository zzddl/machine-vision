#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/12
# @file 图像基本操作.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
# opencv的读取顺序为B, G, R
img = cv2.imread('yellow.jpg')
h, w, c = img.shape
print(h, w, c)  #　ｃ是通道数目
# 访问某个像素的像素值，返回数组中包含BGR
print(img[100, 100])

# 仅访问蓝色通道的像素
blue = img[100, 100, 0]
print(blue)

# 访问红色通道的值
red = img.item(10, 10, 2)
print(red)
# 修改红色通道的值
img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))

img1 = cv2.imread('roi.jpg')
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
replicate = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT101)
wrap = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
# 需要指定value值
constant = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')

plt.show()


