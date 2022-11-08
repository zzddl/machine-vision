#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/31
# @file main.py

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
classnames = []
# 包含了很多目标标签，按照一定顺序排列，基本包含了Yolo官方模型中可检测的对象
classfile = 'coco.names'

with open(classfile, 'rt') as f:
    # rstrip() 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符。
    # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)

# 这都是官方提供的
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
        cv2.putText(img, classnames[classId-1].upper(), (box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)
