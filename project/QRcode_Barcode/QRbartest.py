#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/26
# @file QRbartest.py

from pyzbar.pyzbar import decode
import cv2
import numpy as np

# img = cv2.imread('zhangdinglong.png')

# 获取摄像头资源
capture = cv2.VideoCapture(0)

with open('MyDataFile.txt') as f:
    MyDataList = f.read().splitlines()  # 读取文件内容，没有\n
print(MyDataList)

while True:

    success, img = capture.read()

    for barcode in decode(img):
        # barcode中包含data（码中存储的信息），type（码的类型），rect（左上角坐标和宽、高），polygon（外界多边形框的四个顶点的坐标）
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
        print(pts)
        pts = pts.reshape((-1, 1, 2))  # -1表示自动计算，shape为(4, 1, 2)。导入polylines之前都要做这个操作(-1,1,2)
        print([pts])
        cv2.polylines(img, [pts], True, mycolor, 4)
        pts2 = barcode.rect  # barcode的外界矩形
        print(pts2)
        # (pts2[0], pts2[1])是左上角顶点的坐标。0.9是字体大小
        cv2.putText(img, output, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mycolor, 2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
