#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/26
# @file test.py

import numpy as np
import cv2
pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
pts = pts.reshape(-1, 1, 2) #只画出四边形的点   相当于把一个点作为一个四边形画
print(pts)
pts = pts.reshape(-1, 4, 2) #画出了四边形   一个四边形需要四个点 所以第二维应该为4 包含四个点的坐标，第三维为2 表示坐标含两个元素x，y
print(pts)
# print(a.sum(axis=1))
# print(np.diff(a, axis=1))