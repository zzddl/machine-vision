#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GradientDescent.py
@Time    :   2022/11/09 09:55:54
@Author  :   Dinglong Zhang 
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
'''

# here put the import lib
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
cost_list = []

# 初始权重的猜测
w = 1.0

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

# 计算梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print('predict (before training):', 4, forward(4))


for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    cost_list.append(cost_val)
    print('epoch=',epoch, 'w=',w, 'cost=',cost_val)
print('predict (after training):',4,forward(4))

x = list(range(0,100))

plt.plot(x,cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()