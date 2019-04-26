import getpass
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from sklearn.metrics import roc_curve, auc
import copy
import torchvision.models as models
from data_loader import BreastPathQDataSet

def main():
    """
    Main Function.
    :return:
    """
    define_gpu()
    train_data = BreastPathQDataSet(split="train")
    val_data = BreastPathQDataSet(split="val")
    test_data = BreastPathQDataSet(split="test")


def define_gpu(minimum_memory_mb=1800):
    gpu_to_use = 0
    try:
        print('GPU already assigned before: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        return
    except:
        pass
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)
    print('Chosen GPU: ' + str(0))
    x = torch.rand((256,1024,minimum_memory_mb-500)).cuda()
    del x
    x = torch.rand((1,1)).cuda()
    del x


main()
