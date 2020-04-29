# -*- coding:utf-8 -*-
"""
@file name  : test_tensorboard.py
@copyright  : TingsongYu https://github.com/TingsongYu
@author     : perry
@date       : 2020-04-28
@brief      : 测试tensorboard可正常使用
"""
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 记录可视化的数据
writer = SummaryWriter(comment='test_tensorboard')

for x in range(100):

    # 分别演示记录y=2x、y=pow(2, x)和xsinx、xcosx、arctanx函数
    writer.add_scalar('y=2x', x * 2, x)
    writer.add_scalar('y=pow(2, x)',  2 ** x, x)
    
    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()

