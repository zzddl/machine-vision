#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/27
# @file HSV目标追踪.py

import cv2
import numpy as np

cap = cv2.VideoCapture('orange1.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v') #设置输出视频格式
fps =cap.get(cv2.CAP_PROP_FPS) #设置输出视频帧数
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) #视频尺寸


out2 = cv2.VideoWriter('orange_hsv.mp4',fourcc, fps, size) #设置输出hsv视频
out3 = cv2.VideoWriter('orange_mask.mp4',fourcc, fps, size) #设置输出mask视频
out4 = cv2.VideoWriter('orange_res.mp4',fourcc, fps, size) #设置输出最终过滤视频
out5 = cv2.VideoWriter('orange_result.mp4',fourcc, fps, size) #设置输出mask视频


while cap.isOpened():
    ret, frame = cap.read()

    if ret==True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([11, 43, 46])
        upper = np.array([25, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)

        res = cv2.bitwise_and(frame, frame, mask=mask2)

        kernel = np.ones((10, 10), np.uint8)  # 设置开运算所需核
        opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)  # 对得到的mask进行开运算
        print(opening)
        rectangle = np.where(opening == 255)  # 找出开运算后矩阵中为255的部分，即物体范围
        cv2.rectangle(frame, (min(rectangle[1]), min(rectangle[0])), (max(rectangle[1]), max(rectangle[0])),
                      (0, 0, 255), 3)  # 根据每一帧中物体的左上角坐标以及右下角坐标绘制矩形框
        out2.write(hsv)  # 保存hsv视频到本地
        out3.write(mask2)  # 保存mask视频到本地
        out4.write(res)  # 保存最终视频到本地
        out5.write(frame)
        cv2.imshow('frame', res)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
