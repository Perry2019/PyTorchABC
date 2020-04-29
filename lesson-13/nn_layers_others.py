# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_others.py
# @copyright  : tingsongyu
# @author     : perry
# @date       : 2019-04-26
# @brief      : 池化、线性、激活层
"""
import os
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
from tools.common_tools import transform_invert, set_seed

set_seed(1)  # 设置随机种子

# ================================= load img ==================================
# 读取RGB图像
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor 转换成张量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ maxpool 最大值池化 ==========================================================
# flag = 1
flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))   # 池化核尺寸:2*2   步长：2*2
    img_pool = maxpool_layer(img_tensor)

# ================ avgpool 平均池化 ============================================================
# flag = 1
flag = 0
if flag:
    avgpoollayer = nn.AvgPool2d((2, 2), stride=(2, 2))   # 池化核尺寸:2*2   步长：2*2
    img_pool = avgpoollayer(img_tensor)

# ================ avgpool divisor_override 因子的使用 ========================================
# flag = 1
flag = 0
if flag:
    # 此处不是调用图像而是自己初始化产生一个4*4的张量
    img_tensor = torch.ones((1, 1, 4, 4))
    # divisor_override=1 等价于平均池化
    # avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=1)

    # divisor_override=3
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print("raw_img:\n{}\npooling_img:\n{}".format(img_tensor, img_pool))


# ================ max unpool 反池化 ============================================
# flag = 1
flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)  # 随机初始化4*4图像
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)  # 最大化索引
    img_pool, indices = maxpool_layer(img_tensor)                             # 记录最大化池化对应的索引

    # unpooling  反池化
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)          # 创建随机的反池化的输入
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))                  # 构建反池化层
    img_unpool = maxunpool_layer(img_reconstruct, indices)                   # 将创建的随机输入和索引传入反池化层

    print("raw_img:\n{}\nimg_pool:\n{}".format(img_tensor, img_pool))
    print("img_reconstruct:\n{}\nimg_unpool:\n{}".format(img_reconstruct, img_unpool))


# ================ linear 全连接层/线性层（未考虑激活函数）==================================================
flag = 1
# flag = 0
if flag:
    inputs = torch.tensor([[1., 2, 3]])                      # 输入
    linear_layer = nn.Linear(3, 4)                           # 线性层三个输入节点，四个输出节点
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],   # 权值矩阵
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])

    linear_layer.bias.data.fill_(0.5)                         # 偏置
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)


# ================================= visualization ==================================
# print("池化前尺寸:{}\n池化后尺寸:{}".format(img_tensor.shape, img_pool.shape))
# img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()











