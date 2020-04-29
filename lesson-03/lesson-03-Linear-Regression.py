# -*- coding:utf-8 -*-
"""
@file name  : lesson-03-Linear-Regression.py
@copyright  : tingsongyu
@author     : Perry
@date       : 2020-04-23
@brief      : 一元线性回归模型
"""
import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

# 学习率    20191015修改
lr = 0.05

# 创建训练数据
x = torch.rand(20, 1) * 10  # x data (tensor), shape=(20, 1)
y = 2*x + (5 + torch.randn(20, 1))  # y data (tensor), shape=(20, 1), torch.randn(20, 1)是添加的随机噪声

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):

    # 前向传播(构建y=wx+b)
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss(0.5是为了约掉求导时的2)
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播（计算反向传播的梯度）
    loss.backward()

    # 更新参数(w=w-LR*w.grad  b=b-LR*w.grad)
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 清零张量的梯度(20191015增加,每次迭代完一次对张量清零)
    w.grad.zero_()
    b.grad.zero_()

    # 绘图
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy()) # 训练数据的散点图
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)  # 每次迭代得到的拟合线
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'}) # 显示损失
        plt.xlim(1.5, 10)  # X轴
        plt.ylim(8, 28)    # Y轴
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy())) # 表头数据
        plt.pause(1)     # 动图显示的间隔

        # 停止的条件，当loss小于1时停止训练
        if loss.data.numpy() < 1:
            break
