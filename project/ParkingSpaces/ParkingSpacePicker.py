#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ParkingSpacePicker.py
@Time    :   2022/11/10 16:18:35
@Author  :   Dinglong Zhang 
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
'''

# here put the import lib
import cv2
import pickle

img = cv2.imread('carParkImg.png')

try:
    with open('CarParkPos', 'rb') as f:
        # load()读取指定的序列化数据文件，并返回对象
        posList = pickle.load(f)
except:
    # 没能正确读取文件的时候，就创建一个新的列表
    posList = []

width, height = 107, 48


def MouseClick(events, x, y, flags, params):
    """
    这里必须传入五个参数
    :param events:
    :param x: 鼠标点击的x坐标
    :param y: 鼠标点击的y坐标
    :param flags:
    :param params:
    :return:
    """
    # 左键创建矩形框
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    # 右键删除
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            # 获取矩形框的左上角的位置坐标
            x1, y1 = pos
            # 判断鼠标点击的位置是不是在已经存在的矩形框当中
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    # pickle中dump()要求必须是以'wb'的打开方式进行操作
    with open('CarParkPos', 'wb') as f:
        # dump()方法将数组对象存储为二进制的形式，并写入文件
        pickle.dump(posList, f)


while True:
    # 每次都要导入一次图片，否则删除操作无法进行
    img = cv2.imread('carParkImg.png')

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', MouseClick)
    cv2.waitKey(1)
