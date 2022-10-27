#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Dinglong Zhang
# @date 2022/10/26
# @file test.py

import numpy as np
a = np.array([[[0, 3], [5, 8], [3, 9], [2, 10]]])
print(a)
print(a.shape)
print(a.reshape(2, 4))
b = a.reshape(-1, 1, 2)
print(b)
print(b.shape)
print([b])
# print(a.sum(axis=1))
# print(np.diff(a, axis=1))