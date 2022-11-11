#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2022/11/10 16:15:27
@Author  :   Dinglong Zhang
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
"""

# here put the import lib
import cv2
import cvzone
import pickle
import numpy as np

cap = cv2.VideoCapture('carpark.avi')

with open('CarParkPos', 'rb') as f:
    # load()读取指定的序列化数据文件，并返回对象
    posList = pickle.load(f)

width, height = 107, 48


def checkParkingSpaces(imgpro):
    # 统计停车位数量
    SpaceCounter = 0
    for pos in posList:
        x, y = pos
        # 遍历所有的车位
        imgCrop = imgpro[y:y + height, x:x + width]
        # cv2.imshow(str(x*y), imgCrop)
        # 返回灰度值不为0的像素数，像素数少的说明轮廓少，则没车
        count = cv2.countNonZero(imgCrop)

        if count < 800:
            color = (0, 255, 0)
            thickness = 5
            SpaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 35), scale=1.2,
                           thickness=2, offset=0, colorR=color)
        # 把空余的车位数量显示在左上角
        cvzone.putTextRect(img, f'Free:{SpaceCounter}/{len(posList)}', (0,50), scale=3,
                           thickness=5, offset=20, colorR=(0, 255, 0))


while True:
    # 将视频循环播放
    # 如果视频的当前的帧的位置==视频的总帧数
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # 设置当前帧为起始帧的位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    # 获取灰度图
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波,对原始图像进行平滑操作
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # 获取二值图像
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    # 中值滤波，减少二值图像当中的噪声
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    # 膨胀操作，对边界扩展，更方便区分是否有车
    kernel = np.ones((3, 3), np.int8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    checkParkingSpaces(imgDilate)

    cv2.imshow('Image', img)
    # cv2.imshow('ImageBlur', imgBlur)
    # cv2.imshow('ImageThreshold', imgThreshold)
    # cv2.imshow('ImageMedian', imgMedian)
    # cv2.imshow('ImageDilate', imgDilate)

    cv2.waitKey(1)

    