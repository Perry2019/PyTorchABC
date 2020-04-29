# -*- coding:utf-8 -*-
"""
@file name  : lr_decay_scheduler.py
@copyright  : TingsongYu https://github.com/TingsongYu
@author     : perry
@date       : 2020-04-28
@brief      : 学习率下降策略
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)

LR = 0.1           # 学习率
iteration = 10     # 迭代次数
max_epoch = 200    # epoch
# ------------------------------ fake data and optimizer  ------------------------------

weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1))

optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# ------------------------------ 1 Step LR ------------------------------
# 等间隔调整学习率（间隔以epoch为衡量标准）
flag = 0
# flag = 1
if flag:
    # 设置学习率下降策略（每50个epoch调整一次学习率）
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):                    # epoch循环，即执行200次epoch

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):                    # iteration循环，即每个epoch的数据执行iteration次梯度计算

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 2 Multi Step LR ------------------------------
# 按自定义的给定间隔调整学习率（间隔以epoch为衡量标准）
flag = 0
# flag = 1
if flag:

    milestones = [50, 125, 160]     # 调整学习率的间隔，第50个，125个和160个epoch调整学习率
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 3 Exponential LR ------------------------------
# 按指数衰减调整学习率
flag = 0
# flag = 1
if flag:

    gamma = 0.95
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Exponential LR Scheduler\ngamma:{}".format(gamma))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# ------------------------------ 4 Cosine Annealing LR ------------------------------
flag = 0
# flag = 1
if flag:

    t_max = 50
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):

            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ------------------------------ 5 Reduce LR On Plateau ------------------------------
# 设定监控指标调整学习率
# flag = 0
flag = 1
if flag:
    loss_value = 0.5
    accuray = 0.9

    factor = 0.1
    mode = "min"
    patience = 10
    cooldown = 10
    min_lr = 1e-4
    verbose = True

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                        cooldown=cooldown, min_lr=min_lr, verbose=verbose)

    for epoch in range(max_epoch):
        for i in range(iteration):

            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        # if epoch == 5:
        #     loss_value = 0.4

        scheduler_lr.step(loss_value)   # 监控变量loss_value


# ------------------------------ 6 lambda ------------------------------
# 自定义调整策略（可对不同的参数组设定不同的学习率调整方法）
flag = 0
# flag = 1
if flag:

    lr_init = 0.1

    # 设定两个参数组
    weights_1 = torch.randn((6, 3, 5, 5))
    weights_2 = torch.ones((5, 5))

    # 含有两个参数组的优化器
    optimizer = optim.SGD([
        {'params': [weights_1]},
        {'params': [weights_2]}], lr=lr_init)

    # 两个参数组的学习率调整策略
    lambda1 = lambda epoch: 0.1 ** (epoch // 20)
    lambda2 = lambda epoch: 0.95 ** epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        for i in range(iteration):

            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        lr_list.append(scheduler.get_lr())
        epoch_list.append(epoch)

        print('epoch:{:5d}, lr:{}'.format(epoch, scheduler.get_lr()))

    plt.plot(epoch_list, [i[0] for i in lr_list], label="lambda 1")
    plt.plot(epoch_list, [i[1] for i in lr_list], label="lambda 2")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LambdaLR")
    plt.legend()
    plt.show()
