# -*- coding: utf-8 -*-
"""
@file name  : lesson-05-autograd.py
@copyright  : tingsongyu
@author     : Perry
@date       : 2020-04-23
@brief      : 自动求导
"""
import torch
torch.manual_seed(10)


# ====================================== retain_graph ==============================================
# torch.autograd.backward函数中的retain_graph演示（使用lesson4中的计算图）

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 此处使用了类方法，该函数的内部调用了torch.autograd.backward
    y.backward()
    print(w.grad)

    # 验证retain_graph
    # 若要执行多次反向传播，要设置retain_graph=True,使得用于计算梯度的计算图保留
    #  y.backward(retain_graph=True)
    #  y.backward()
    #  print(w.grad)

# ====================================== grad_tensors ==============================================
# torch.autograd.backward函数中的多个参数求导演示

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)    dy0/dw = 5
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # 损失[y0, y1]是向量
    grad_tensors = torch.tensor([1., 2.])   # 多个梯度计算时，设置梯度的权重，y0的梯度权重为1，y1的梯度权重为2 dy0/dw*1+dy1/dw*2

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)


# ====================================== autograd.gard ==============================================
# torch.autograd.grad函数进行高阶求导，高阶求导时create_graph=True

# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


# ====================================== tips: 1 ==============================================
# 自动求导不会自动清零

# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(2):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        # 设置每次执行完梯度计算进行求导清零
        w.grad.zero_()


# ====================================== tips: 2 ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 ==============================================
# flag = True
flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    # a = a + torch.ones((1, ))
    # print(id(a), a)

    a += torch.ones((1, ))
    print(id(a), a)


flag = True
# flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # w.add_(1)

    y.backward()





