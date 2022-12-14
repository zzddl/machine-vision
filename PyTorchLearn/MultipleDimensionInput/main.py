#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/11/15
# @file main.py

import torch
import numpy as np
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵,如果不加[]，得到的是一个向量


# design using model class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 三层神经网络
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Softplus()  # 将其看做是神经网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y_hat
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(7000):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

# 查看某一层的参数
layer1_weight = model.linear1.weight.data
layer1_bias = model.linear1.bias.data
print('layer1_weight', layer1_weight)
print('layer1_weight.shape', layer1_weight.shape)
print('layer1_bias', layer1_bias)
print('layer1_bias.shape', layer1_bias.shape)

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
