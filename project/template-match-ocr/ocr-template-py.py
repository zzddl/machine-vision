#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/17
# @file ocr-template-py.py

import cv2
import numpy as np
import myutils
from imutils import contours

# 　指定信用卡类型
FIRST_NUMBER = {
    "3": "Amercian Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取一个模板图像
img = cv2.imread('./images/ocr_a_reference.png')
cv_show('img', img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白图像（不是灰度图）
# cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中的每个元素都是图像中的一个轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)

print(np.array(refCnts).shape)  # 一共有10个轮廓
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适的大小
    (x, y, w, h) = cv2.boundingRect(c)
    # 获取感兴趣的区域
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每一个数字对应一个模板
    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，并且进行预处理
image = cv2.imread('./images/credit_card_01.png')
cv_show('image', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 礼帽操作（原图像-开运算后的图像），可以得到原图像中的噪声，突出更加明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# Sobel算子
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # -1是按照（3*3）来进行计算
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX', gradX)

# 开操作把通不过的都断开，闭操作把进不去的都填上

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX', gradX)
# THRESH_OTSU(大津二值化算法)会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh', thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cur_image = image.copy()
cv2.drawContours(cur_image, threshCnts, -1, (0, 0, 255), 3)
cv_show('img', cur_image)

# 用来存储外接矩形
locs = []
# 遍历轮廓
for (i, c) in enumerate(threshCnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 筛选合适的区域留下来
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            # 符合条件的留下来
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x:x[0])  # 利用key来进行排序，key为每个x的第一个元素

# 用来储存最后结果
output = []
# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    GroupOutput = []

    # 根据坐标来提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)

    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)

    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():  # digits里面保存的是10个模板
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            print(result)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 找到最合适的数字
        GroupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(GroupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(GroupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
