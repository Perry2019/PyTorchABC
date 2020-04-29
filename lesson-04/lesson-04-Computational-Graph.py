# -*- coding:utf-8 -*-
"""
@file name  : lesson-04-Computational-Graph.py
@copyright  : tingsongyu
@author     : Perry
@date       : 2020-04-23
@brief      : 计算图示例
"""
import torch

# 配合PPT/笔记计算图模型
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

# x,w输入的数据, a=x+w, b=w+1, y=a*b
a = torch.add(w, x)     # retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# 查看叶子结点（用户输入节点时叶子节点）
# print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
# print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
# print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)

