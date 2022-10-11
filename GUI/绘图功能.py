#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/11
# @file 绘图功能.py

import cv2
import numpy as np
img = np.zeros((512, 512, 3), np.uint8)  #512指的是像素高和宽，3指的是BGR的三种颜色

cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)  # 左上角，右下角，颜色，线宽
cv2.circle(img,(447,63), 63, (0,0,255), -1)  # -1表示填充内部
# (256,256)是圆心位置，100是长轴长度，50是短轴长度，第一个0是椭圆在逆时针方向的旋转角角度
# 0,180是长轴顺时针方向测量的圆弧的起点和终点
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# 多边形，要先定义出每个顶点的坐标
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
# 如果为False，得到是所有点的折线，不是一个闭合的图形
cv2.polylines(img,[pts],True,(0,255,255))
# 给图像加文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)  # 4是字体的大小，2是线条宽度
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
