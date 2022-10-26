#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/18
# @file scan.py

import cv2
import numpy as np


# 寻找原图像的四个坐标点(传入的pts数据的原图像的数据，只是顺序不对，order_points是用来改变顺序)
def order_points(pts):
    print('pts', pts)
    # 一共有四个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序0123找到四个坐标点为左上，右上，右下，左下
    # 计算左上，右下(把x，y坐标相加，最小的是左上，最大是右下)
    s = pts.sum(axis=1)
    print('s', s)
    rect[0] = pts[np.argmin(s)]
    print('rect0', rect[0])
    rect[2] = pts[np.argmax(s)]

    # 计算右上，左下（右上是y-x最小的，左下是y-x最大的）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_points_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    print('rect', rect)
    (tl, tr, br, bl) = rect

    # 取较大的
    # 计算输入的w和h的值
    widthA = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    widthB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    maxwidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    heightB = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    maxheight = max(int(heightA), int(heightB))

    # 变换后对应的坐标位置
    dst = np.array([
        [0, 0],
        [maxwidth - 1, 0],
        [maxwidth - 1, maxheight - 1],
        [0, maxheight - 1]],
        dtype='float32'
    )

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)  # 通过原来的四个点和新的四个点来计算变换矩阵
    warped = cv2.warpPerspective(image, M, (maxwidth, maxheight))  # (maxwidth, maxheight)是输出图像的大小

    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 读取图像
image = cv2.imread('./images/receipt.jpg')
print(image.shape)  # (3264, 2448, 3) 3264是height，2448是width

# 坐标也会相同变化
ratio = image.shape[0] / 500
origin = image.copy()

image = resize(origin, height=500)

# 图像预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 利用高斯滤波消除噪声
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny边缘检测
edged = cv2.Canny(gray, 75, 200)

# 展示预处理的结果
print("STEP1:边缘检测")
cv2.imshow('image', image)
cv2.imshow('edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 遍历轮廓
for c in cnts:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)

    # 参数1是源图像的某个轮廓，是一个点集
    # 参数2是是一个距离值，表示多边形的轮廓接近实际轮廓的程度，值越小，得到的多边形角点越多，对原图像的多边形近似效果越好。
    # 参数3表示是否闭合
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 如果是4个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示轮廓的结果
print("STEP2:获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_points_transform(origin, screenCnt.reshape(4, 2) * ratio)  # 按照缩放的比例还原回去

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU)[1]
cv2.imwrite('scan.jpg', ref)

# 展示结果
print("STEP3:变换")
cv2.imshow('Original', resize(origin, height=650))
cv2.imshow('Scanned', resize(ref, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
