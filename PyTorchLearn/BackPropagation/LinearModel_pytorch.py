#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LinearModel_pytorch.py
@Time    :   2022/11/10 10:56:52
@Author  :   Dinglong Zhang 
@Version :   1.0
@Contact :   zhangdinglong@chinaaie.com.cn
@Desc    :   None
'''

# here put the import lib
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

'''
创建一个张量w，并且设置requires_grad为True，程序会追踪所有对于该张量的操作，完成计算后调用backward()，
这个张量所有的梯度都会自动累计到.grad属性当中。
'''
w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)=",4,forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 1.计算loss，前向
        l = loss(x, y)
        # 2.计算偏导并保存在变量中（后向传播）
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        # 3.用梯度下降做更新,注意grad也是一个tensor
        w.data = w.data - 0.01 * w.grad.data
        # 4.在grad更新时，每一次运算后都需要将上一次的梯度记录清空
        # 使用.backward()计算的梯度将会被累加，所以需要清零
        w.grad.data.zero_()

    print('progress:', epoch, l.item())  # 取loss的值时，使用l.item()，因为l是tensor会构建计算图

print("predict (after training)=",4,forward(4).item())
