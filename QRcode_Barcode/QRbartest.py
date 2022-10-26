#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/26
# @file QRbartest.py

from pyzbar.pyzbar import decode
import cv2
import numpy as np

img = cv2.imread('zhangdinglong.png')

# 获取摄像头资源
capture = cv2.VideoCapture(0)

with open('MyDataFile.txt') as f:
    MyDataList = f.read().splitlines()
print(MyDataList)


while True:

    success, img = capture.read()

    for barcode in decode(img):
        # print(barcode.data)  # b代表的是byte，
        mydata = barcode.data.decode('utf-8')
        print(mydata)

        if mydata in MyDataList:
            output = 'authorized'
            mycolor = (0, 255, 0)
        else:
            output = 'un-authorized'
            mycolor = (0, 0, 255)

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, mycolor, 4)
        pts2 = barcode.rect
        cv2.putText(img, output, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mycolor, 2)

    cv2.imshow('result', img)
    cv2.waitKey(1)


