#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/24
# @file line-detect.py


import cv2
import numpy as np


def get_edge_img(img, canny_threshold1=50, canny_threshold2=100):
    """
    灰度化，canny变换，提取边缘
    :param img: 彩色图
    :param canny_threshold1:
    :param canny_threshold2:
    :return:
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


def ROI_mask(gray_img):
    """
    对gary_img进行掩膜
    :param gray_img:
    :return:
    """
    poly_pts = np.array([[[0, 368], [300, 210], [340, 210], [640, 368]]])
    mask = np.zeros_like(gray_img)
    mask = cv2.fillPoly(mask, pts=poly_pts, color=255)
    img_mask = cv2.bitwise_and(gray_img, mask)
    return img_mask


def get_lines(edge_img):
    """
    获取edge_img中的所有的线段
    :param edge_img:标记边缘的灰度图
    :return:
    """

    def calculate_slope(line):
        """
        计算线段line的斜率
        :param line: np.array([x_1, y_1, x_2, y_2])
        :return:
        """
        x_1, y_1, x_2, y_2 = line[0]
        return (y_2 - y_1) / (x_2 - x_1)

    def reject_abnormal_line(lines, threshold=0.2):
        """
        剔除斜率不一致的线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        :param threshold: 斜率阈值,如果差值大于阈值，则剔除
        :return:
        """

        slope = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slope)
            diff = [abs(s - mean) for s in slope]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slope.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines

    def least_squares_fit(lines):
        """
        将lines中的线段拟合成一条线段
        :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
        :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
        """

        # 获取所有的x,y值，转化为一维的数组
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])

        poly = np.polyfit(x_coords, y_coords, deg=1)  # 曲线拟合
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))

        return np.array([point_min, point_max], dtype=np.int)

    # 进行霍夫变换获取所有的直线
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 15, minLineLength=40, maxLineGap=20)

    # 按照斜率区分车道线
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]

    # 剔除离群线段
    left_lines = reject_abnormal_line(left_lines)
    right_lines = reject_abnormal_line(right_lines)

    return least_squares_fit(left_lines), least_squares_fit(right_lines)


def draw_lines(img, lines):
    """
    在img上面绘制lines
    :param img:
    :param lines: 两条线段: [np.array([[xmin1, ymin1], [xmax1, ymax1]]), np.array([[xmin2, ymin2], [xmax2, ymax2]])]
    :return:
    """
    left_line, right_line = lines
    cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 0, 255), thickness=5)

    cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 0, 255), thickness=5)


def show_line(color_img):
    """
    在color_img上面画出车道线
    :param color_img:
    :return:
    """
    edge_img = get_edge_img(color_img)
    mask_gray_img = ROI_mask(edge_img)
    lines = get_lines(mask_gray_img)
    draw_lines(color_img, lines)
    return color_img


# 识别图片
color_img = cv2.imread('img.jpg')
result = show_line(color_img)
cv2.imshow('output', result)
cv2.waitKey(0)

# 识别视频
#
# capture = cv2.VideoCapture('video.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# outfile = cv2.VideoWriter('output.avi', fourcc, 25., (1280, 368))
# # 循环处理每一帧视频
# while capture.isOpened():
#     _, frame = capture.read()
#     origin = np.copy(frame)
#     frame = show_line(frame)
#     output = np.concatenate((origin, frame), axis=1)
#     outfile.write(output)
#     cv2.imshow('output', frame)
#     # 处理退出
#     if cv2.waitKey(1) == ord('q'):
#         cv2.destroyAllWindows()
#         break
