import torch
import numpy as np
import random
import os


def init_random_seed(seed=42):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU. torch的GPU随机性，为所有GPU设置随机种子

    # 是否启用cuDNN加速，默认为True，如果设置为false，则下面两个设置将不起作用，通常是使用CPU进行计算
    # 这可能会导致计算速度较慢，但可以提供确定性的结果，而不受 cuDNN 特定配置和优化的影响
    torch.backends.cudnn.enabled = True

    # benchmark和deterministic搭配使用，一般为相反设置。true+false=faster and less reproducible
    # false + true = slower and more reproducible(多用于复现)，默认为true+false
    torch.backends.cudnn.benchmark = True  # 让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    torch.backends.cudnn.deterministic = False  # 若为true，则每次返回算法确定(默认算法)，即固定了每层的算法
