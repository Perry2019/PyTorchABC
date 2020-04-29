# -*- coding: utf-8 -*-
"""
# @file name  : grad_vanish_explod.py
# @copyright  : tingsongyu
# @author     : perry
# @date       : 2019-04-27
# @brief      : 梯度消失与爆炸实验
"""
import os
import torch
import random
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)

            # 非线性激活函数
            x = torch.relu(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 未考虑进行初始化权值
                # nn.init.normal_(m.weight.data)    # normal: mean=0, std=1 初始化

                # 进行初始化权值（未考虑激活函数，只将每个输出层的方差设为1）
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))    # normal: mean=0, std=1

                # Xavier初始化（手写）
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                # nn.init.uniform_(m.weight.data, -a, a)

                # Xavier初始化
                # tanh_gain = nn.init.calculate_gain('tanh')
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                # Kingming初始化（手写）
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))

                # Kingming初始化（调用）
                nn.init.kaiming_normal_(m.weight.data)

# flag = 0
flag = 1

if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # 输入数据0均值1标准差 normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================
# 激活函数的方差变化尺度：输入数据的方差除以输出数据的方差

flag = 0
# flag = 1

if flag:

    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)


























