# -*- coding:utf-8 -*-
"""
@file name  : dropout_layer.py
@copyright  : TingsongYu https://github.com/TingsongYu
@author     : perry
@date       : 2020-04-29
@brief      : 验证dropout在训练集上实施数据的尺度变化，以避免测试集上的变化
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tools.common_tools import set_seed
from torch.utils.tensorboard import SummaryWriter

# set_seed(1)  # 设置随机种子，该值打开后，不同次的操作均使用相同的随机数

# 定义网络类
class Net(nn.Module):
    def __init__(self, neural_num, d_prob=0.5):
        super(Net, self).__init__()

        self.linears = nn.Sequential(

            nn.Dropout(d_prob),                   # 输入层设置droupout
            nn.Linear(neural_num, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linears(x)

input_num = 10000
x = torch.ones((input_num, ), dtype=torch.float32)

net = Net(input_num, d_prob=0.5)                  # 调用开始定义的class Net(nn.Module):
net.linears[1].weight.detach().fill_(1.)          # 权重均设置为1

net.train()                                       # 训练模式下计算前向传播
y = net(x)
print("output in training mode", y)

net.eval()                                        # 测试模式下计算前向传播
y = net(x)
print("output in eval mode", y)

















