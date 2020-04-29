# -*- coding: utf-8 -*-
"""
# @file name  : optimizer_methods.py
# @copyright  : TingsongYu https://github.com/TingsongYu
# @author     : perry
# @date       : 2020-04-27
# @brief      : optimizer's methods
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.optim as optim
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子

weight = torch.randn((2, 2), requires_grad=True)         # 构建随机权值矩阵
weight.grad = torch.ones((2, 2))                         # 构建全1的梯度

optimizer = optim.SGD([weight], lr=1)

# ----------------------------------- step -----------------------------------
# 优化器的一步更新optimizer.step()使用
flag = 0
# flag = 1
if flag:
    print("weight before step:{}".format(weight.data))
    optimizer.step()                                     # 优化器进行1步更新
    print("weight after step:{}".format(weight.data))


# ----------------------------------- zero_grad -----------------------------------
# 优化器的清空梯度optimizer.zero_grad()使用
flag = 0
# flag = 1
if flag:

    print("weight before step:{}".format(weight.data))
    optimizer.step()        # 修改lr=1 0.1观察结果
    print("weight after step:{}".format(weight.data))

    print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))

    print("weight.grad is {}\n".format(weight.grad))
    optimizer.zero_grad()                                  # 优化器的清空梯度
    print("after optimizer.zero_grad(), weight.grad is\n{}".format(weight.grad))


# ----------------------------------- add_param_group -----------------------------------
# 优化器增加参数组optimizer.add_param_group()使用，如不同的学习率
flag = 0
# flag = 1
if flag:
    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

    w2 = torch.randn((3, 3), requires_grad=True)

    optimizer.add_param_group({"params": w2, 'lr': 0.0001})    # 优化器增加参数组

    print("optimizer.param_groups is\n{}".format(optimizer.param_groups))

# ----------------------------------- state_dict -----------------------------------
# 优化器获取当前状态信息字典optimizer.state_dict()的使用
flag = 0
# flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n", opt_state_dict)

    for i in range(10):                                         # 优化器更新10次后保存的状态信息
        optimizer.step()

    print("state_dict after step:\n", optimizer.state_dict())

    # 优化器状态信息保存为pkl文件
    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

# -----------------------------------load state_dict -----------------------------------
# 加载优化器停止前保存的状态字典optimizer.load_state_dict()的使用
# flag = 0
flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))   # 读取之前保存的优化器的状态信息

    print("state_dict before load state:\n", optimizer.state_dict())
    optimizer.load_state_dict(state_dict)
    print("state_dict after load state:\n", optimizer.state_dict())












