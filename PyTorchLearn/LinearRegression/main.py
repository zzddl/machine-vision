#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/11/14
# @file main.py


import torch
import matplotlib.pyplot as plt

# 1.prepare dataset
# x,y是矩阵，3行1列，一共有三个数据，每个数据有1个特征
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# 2.design model using class
'''
Our model class should be inherit from nn.Module, which is Base class for all neural network modules.
Class nn.Linear contain two member Tensors: weight and bias.
Class nn.Linear has implemented the magic method __call__(), which enable the instance of the class 
can be called just like a function. Normally the forward()will be called.
'''


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是linear.weight/linear.bias
        # 第三个参数默认为b，表示用到b
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)  # 计算y = wx + b
        return y_pred


model = LinearModel()

# 3.Construct Loss and Optimizer
criterion = torch.nn.MSELoss(size_average=False)
# model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

# 4.training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)  # Forward: Predict，调用model.forward(x_data)函数
    loss = criterion(y_pred, y_data)  # Forward: Loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # The grad computed by .backward() will be accumulated.So before backward, remember set the grad to ZERO!!!
    loss.backward()  # Backward: Autograd，自动计算梯度
    optimizer.step()  # Update 更新w和b的值

    epoch_list.append(epoch)
    loss_list.append(loss.item())

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_test = ', y_test.data)

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
