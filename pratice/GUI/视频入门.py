#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/11
# @file 视频入门.py

import cv2
import numpy as np
# 调用电脑的摄像头
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # FourCC 是用于指定视频解码器的 4 字节代码
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 20.0为fps
while(cap.isOpened()):
    # 一帧一帧的捕捉
    ret, frame = cap.read() # 返回一个bool值，如果加载成功返回True
    if ret == True:
        # 翻转图像
        frame = cv2.flip(frame, 0)
        out.write(frame)
        # 显示返回的每一帧
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# 释放VideoCapture
cap.release()
out.release()
cv2.destroyAllWindows()