import getpass
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import copy
import torchvision.models as models
from data_loader import BreastPathQDataSet
from basic_conv import BaselineConvNet
from utils import Utils


def main():
    """
    Main Function.
    :return:
    """
    define_gpu()
    train_data = BreastPathQDataSet(split="train")
    val_data = BreastPathQDataSet(split="val")
    test_data = BreastPathQDataSet(split="test")

    epochs = [5, 50, 100]
    learning_rates = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    batch_size = [4, 8, 16, 32]

    # basic_model = Utils(train_data, val_data, test_data, BaselineConvNet())
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=1)
    basic_model = Utils(train_data, val_data, test_data, resnet18)

    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MSELoss()

    for epoch in epochs:
        for batch_size in batch_size:
            for lr in learning_rates:
                # optimizer = torch.optim.SGD(basic_model.parameters(), lr=lr, momentum=0.9, nesterov=True)
                optimizer = torch.optim.Adam(basic_model.parameters(), lr=lr)
                print("Max Epochs:", epoch, "Learning Rate:", lr, "Batch Size:", batch_size)
                trained_model = basic_model.train(epoch, batch_size, criterion, optimizer)


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
    x = torch.rand((1, 1)).cuda()
    del x


main()
