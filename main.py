import getpass
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    gpu_mem = 1800
    if len(sys.argv) > 1 and sys.argv[1] == 'cade':
        print("GPU Mem Setting:", sys.argv[1])
        gpu_mem = 3800
    define_gpu(minimum_memory_mb=gpu_mem)

    train_data = BreastPathQDataSet(split="train")
    val_data = BreastPathQDataSet(split="val")
    test_data = BreastPathQDataSet(split="test")

    learning_rates = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    batch_size = [4, 8, 16, 32]


    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()

    # for epoch in epochs:
    #     for batch_size in batch_size:
    #         for lr in learning_rates:
    #             optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #             if len(sys.argv) > 3:
    #                 if sys.argv[3] == 'sgd':
    #                     optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    #             print("Max Epochs:", epoch, "Learning Rate:", lr, "Batch Size:", batch_size)
    #             trained_model = model.train(epoch, batch_size, criterion, optimizer)

    losses_figure = plt.figure()
    losses_figure_ax = losses_figure.add_subplot(111)

    scores_figure = plt.figure()
    scores_figure_ax = scores_figure.add_subplot(111)

    epochs = 10

    for lr in learning_rates:
        if len(sys.argv) > 2:
            if sys.argv[2] == 'resnet':
                print("Running ResNet")
                resnet = models.resnet18(pretrained=True)
                resnet.fc = torch.nn.Linear(in_features=512, out_features=1)
                utils = Utils(train_data, val_data, test_data)
                model = resnet
            elif sys.argv[2] == 'vgg':
                print("Running VGG")
                vgg = models.vgg11_bn(pretrained=True)
                vgg_modules = list(vgg.children())
                vgg_modules.append(torch.nn.Linear(in_features=1000, out_features=1))
                vgg = torch.nn.Sequential(*vgg_modules)
                utils = Utils(train_data, val_data, test_data)
                model = vgg
            else:
                utils = Utils(train_data, val_data, test_data)
                model = BaselineConvNet()
        else:
            utils = Utils(train_data, val_data, test_data)
            model = BaselineConvNet()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if len(sys.argv) > 3:
            if sys.argv[3] == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

        print("Learning Rate:", lr)
        trained_model, losses, scores = utils.train(model, epochs, 4, criterion, optimizer)
        
        label = "Learning Rate: " + str(lr)
        losses_figure_ax.plot(range(0, epochs), losses, label=label)
        scores_figure_ax.plot(range(0, epochs), scores, label=label)

    losses_figure_ax.set_title("Losses vs. Epochs")
    losses_figure_ax.set_xlabel("Epochs")
    losses_figure_ax.set_ylabel("Losses")
    losses_figure_ax.legend()
    losses_figure.savefig("Losses_150e.png")

    scores_figure_ax.set_title("Scores vs. Epochs")
    scores_figure_ax.set_xlabel("Epochs")
    scores_figure_ax.set_ylabel("Losses")
    scores_figure_ax.legend()
    scores_figure.savefig("Scores_150e.png")


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
    x = torch.rand((256, 1024, minimum_memory_mb-500)).cuda()
    del x
    x = torch.rand((1, 1)).cuda()
    del x


main()
