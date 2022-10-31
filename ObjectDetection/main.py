#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/31
# @file main.py

import cv2
import numpy as np

# img = cv2.imread('lena.png')
cap = cv2.VideoCapture(0)


classnames = []
classfile = 'coco.names'

with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)

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
        cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 150, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(1)
