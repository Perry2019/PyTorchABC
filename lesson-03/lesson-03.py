# -*- coding:utf-8 -*-
"""
@file name  : lesson-02.py
@copyright  : tingsongyu
@author     : Perry
@date       : 2020-04-23
@brief      : 张量操作
"""

import torch
torch.manual_seed(1)

# ======================================= example 1 =======================================
# 使用torch.cat拼接

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))
    print(t)

    # dim=0 在行方向上拼接，反之在列方向上拼接
    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t], dim=1)

    print("t_0:{} shape:{}\nt_1:{} shape:{}".format(t_0, t_0.shape, t_1, t_1.shape))


# ======================================= example 2 =======================================
# 使用torch.stack拼接

# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))
    print(t)

    t_stack = torch.stack([t, t], dim=2)

    print("\nt_stack:{} shape:{}".format(t_stack, t_stack.shape))


# ======================================= example 3 =======================================
# 使用torch.chunk对张量切分

# flag = True
flag = False

if flag:
    a = torch.ones((2, 7))  # 7
    print(a)

    list_of_tensors = torch.chunk(a, dim=1, chunks=3)   # 3

    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))


# ======================================= example 4 =======================================
# 使用torch.split对张量切分

# flag = True
flag = False

if flag:
    t = torch.ones((2, 5))
    print(t)

    list_of_tensors = torch.split(t, [2, 1, 2], dim=1)  # [2 , 1, 2]
    for idx, t in enumerate(list_of_tensors):
        print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))

    # list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
    # for idx, t in enumerate(list_of_tensors):
    #     print("第{}个张量：{}, shape is {}".format(idx, t, t.shape))


# ======================================= example 5 =======================================
# 使用torch.index_select对张量索引

# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)    # float数据类型报错
    t_select = torch.index_select(t, dim=0, index=idx)
    print("t:\n{}\nt_select:\n{}".format(t, t_select))

# ======================================= example 6 =======================================
# 使用torch.masked_select对张量索引

# flag = True
flag = False

if flag:

    t = torch.randint(0, 9, size=(3, 3))
    mask = t.gt(5)  # ge is mean greater than or equal/   gt: greater than/  le:greater than or equal/  lt:
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))


# ======================================= example 7 =======================================
# 使用torch.reshape对张量形状变换

# flag = True
flag = False

if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (-1, 2))    # -1：当torch.reshape(t, (-1,2))时表示第一个维度不用管
    # t_reshape = torch.reshape(t, (-1, 2, 2))
    print("t:{}\nt_reshape:\n{}".format(t, t_reshape))

    # 检查使用reshape占用相同的内存地址
    # t[0] = 1024
    # print("t:{}\nt_reshape:\n{}".format(t, t_reshape))
    # print("t.data 内存地址:{}".format(id(t.data)))
    # print("t_reshape.data 内存地址:{}".format(id(t_reshape.data)))


# ======================================= example 8 =======================================
# 使用torch.transpose对张量形状变换

# flag = True
flag = False

if flag:
    # torch.transpose
    t = torch.rand((2, 3, 4))
    # t的第1维和第2维交换
    t_transpose = torch.transpose(t, dim0=1, dim1=2)    # c*h*w     h*w*c 常用于图像的预处理
    print("t shape:{}\nt_transpose shape: {}".format(t.shape, t_transpose.shape))


# ======================================= example 9 =======================================
# 使用torch.squeeze对张量压缩

# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)
    t_0 = torch.squeeze(t, dim=0)
    t_1 = torch.squeeze(t, dim=1)
    print(t)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)


# ======================================= example 8 =======================================
# 使用torch.add进行数学运算

flag = True
# flag = False

if flag:
    t_0 = torch.randn((3, 3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0, 10, t_1)

    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))














