#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/11/1
# @file NMS.py


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
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    print(confs)
    confs = list(map(float, confs))
    print(confs)

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classnames[classIds[i][0] - 1].upper(), (box[0]+10, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
