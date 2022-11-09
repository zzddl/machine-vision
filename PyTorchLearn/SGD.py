#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SGD.py
@Time    :   2022/11/09 14:02:16
@Author  :   Dinglong Zhang 
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
'''

# here put the import lib

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始权重的猜测
w = 1.0

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x,y):
    return 2 * x * (x * w - y)

print('predict (before training):', 4, forward(4))

# Update weight by every grad of sample of train set.
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad_val = gradient(x,y)
        w -= 0.01 * grad_val
        print('\tgrad:',x,y,grad_val)
        l = loss(x,y)
    print('epoch=',epoch,'\tw=',w,'\tloss=',l)

print('predict (after training):', 4, forward(4))

