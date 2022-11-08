#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/11
# @file 鼠标作为画笔.py

import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:  # 鼠标双击
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

img = np.zeros((512, 512, 3), np.uint8)  #512指的是像素高和宽，3指的是BGR的三种颜色
cv2.namedWindow("image")
cv2.setMouseCallback('image', draw_circle)  # 返回鼠标动作和坐标
while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()