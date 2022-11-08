import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x,w):
    return x * w

def loss_function(x, w, y):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

w_list = []
mse_list = []
for w in np.arange(0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_val_pred = forward(x_val, w)
        loss_val = loss_function(x_val, w, y_val)
        l_sum += loss_val
        print('\t',x_val, y_val, y_val_pred, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()