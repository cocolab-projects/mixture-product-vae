import os
import math
import shutil
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def get_num_interval(k):
    n = 0
    while math.factorial(n) < k:
        n += 1
    return n


def get_fixed_init(n, a, b):
    interval = np.linspace(a, b, n + 1)
    inits = []
    for i in range(0, len(interval) - 1):
        for j in range(i + 1, len(interval)):
            inits.append([interval[i], interval[j]])
    
    return np.array(inits)
