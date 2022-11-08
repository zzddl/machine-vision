#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MSE_3D.py
@Time    :   2022/11/08 17:04:54
@Author  :   Dinglong Zhang 
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
'''

# here put the import lib

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x,w,b):
    return x * w + b

def loss_function(x, w, b, y):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

w = np.arange(0, 4.1, 0.1)
b = np.arange(-2.0, 2.1, 0.1)

w, b = np.meshgrid(w, b)

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_val_pred = forward(x_val, w, b)
    loss_val = loss_function(x_val, w, b, y_val)
    l_sum += loss_val
    print('\t',x_val, y_val, y_val_pred, loss_val)
mse = l_sum / 3

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.xlabel('w')
plt.ylabel('b')

""" 
rstride:行之间的跨度
cstride：列之间的跨度
cmap：切换颜色组合
 """
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()