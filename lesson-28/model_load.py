# -*- coding: utf-8 -*-
"""
@file name  : model_load.py
@copyright  : TingsongYu https://github.com/TingsongYu
@author     : perry
@date       : 2020-04-29
@brief      : 模型的加载
"""
import torch
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed


class LeNet2(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for p in self.parameters():
            p.data.fill_(20191104)


# ================================== load net ===========================
# 加载保存的整个模型
# flag = 1
flag = 0
if flag:

    path_model = "./model.pkl"
    net_load = torch.load(path_model)

    print(net_load)

# ================================== load state_dict ===========================
# 加载保存的模型的参数
flag = 1
# flag = 0
if flag:

    path_state_dict = "./model_state_dict.pkl"
    state_dict_load = torch.load(path_state_dict)

    print(state_dict_load.keys())      # 只观察字典

# ================================== update state_dict ===========================
# 构建一个新的模型，将加载的数据放到新模型中
flag = 1
# flag = 0
if flag:

    # 构建新网络
    net_new = LeNet2(classes=2019)

    print("加载前: ", net_new.features[0].weight[0, ...])

    # 将保存的模型的字典加载到新模型中
    net_new.load_state_dict(state_dict_load)

    print("加载后: ", net_new.features[0].weight[0, ...])





