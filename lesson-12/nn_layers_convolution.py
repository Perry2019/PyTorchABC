# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_convolution.py
# @copyright  : tingsongyu
# @author     : perry
# @date       : 2019-04-26
# @brief      : 学习卷积层
"""
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed

set_seed(4)  # 设置随机种子 （改变随机种子的数字相当于改变随机的卷积层初始权重，输出图像随之改变）

# ================================= load img ==================================
# 读取RGB图像
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor 转换成张量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W 拓展为四维张量，B=Batchsize

# ================================= create convolution layer ==================================

# ================ 2d 创建2维正常卷积 ========================================================
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w) 输入通道3，输出通道1，卷积核为3*3
    nn.init.xavier_normal_(conv_layer.weight.data)  # 采用xavier的方法对上一行创建的卷积层进行初始化

    # calculation  输入图片张量进入卷积层
    img_conv = conv_layer(img_tensor)

# ================ transposed 创建2维正常卷积=================================================
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(i, o, size)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)


# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)  # 对卷积后的张量进行逆操作以显示卷积后的图像
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()



