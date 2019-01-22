# -*- coding:utf-8-*-
# __author__ = 'chen'
# date: 2019/1/22
import numpy as np
import time
# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T      #“.T” 为转置函数。经转置后，该  y  矩阵便包含 4 行 1 列。同我们的输入一致

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1
# a = np.random.random((3,1))
# print(str(a))
First_time = time.time()
for iter in range(10000):       # python3 中取消了 range 函数，而把 xrange 函数重命名为 range，所以现在直接用 range 函数即可
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)
print("Output After Training:")
print(l1)
Last_time = time.time()
print("所用时间: "+str((Last_time-First_time)*1000)+'ms')



