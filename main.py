import getpass
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import copy
import random
import torchvision.models as models
from data_loader import BreastPathQDataSet
from basic_conv import BaselineConvNet
from utils import Utils
from decimal import Decimal


def main():
    """
    Main Function.
    :return:
    """
    machine = "local"
    model_type = "simple"
    optimizer_type = "adam"
    if len(sys.argv) > 1:
        machine = sys.argv[1]
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    if len(sys.argv) > 3:
        optimizer_type = sys.argv[3]

    print("Machine:", machine)
    print("Model:", model_type)
    print("Optimizer:", optimizer_type)
    print()

    if machine == 'local':
        gpu_mem = 1800
    elif machine == 'cade':
        gpu_mem = 3800
    define_gpu(minimum_memory_mb=gpu_mem)

    train_data = BreastPathQDataSet(split="train")
    val_data = BreastPathQDataSet(split="val")
    test_data = BreastPathQDataSet(split="test")

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

    epochs = 20
    learning_rates = [0.01, random.uniform(0.01, 0.0001), random.uniform(0, 0.0001), random.uniform(0, 0.0001), 0.0001]

    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()
    utils = Utils(train_data, val_data, test_data)

    batch_size = [8, 16]
    if optimizer_type == 'sgd':
        batch_size = [16, 32]

    for batch in batch_size:
        for lr in learning_rates:
            if model_type == 'simple':
                model = BaselineConvNet()
            elif model_type == 'resnet':
                resnet = models.resnet18(pretrained=True)
                resnet.fc = torch.nn.Linear(in_features=51200, out_features=1)
                model = resnet

            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

            print("Learning Rate:", lr)
            trained_model, losses, scores = utils.train(model, epochs, batch, criterion, optimizer)
            lr_string = '%.2E' % Decimal(str(lr))
            label = "Learning Rate: " + lr_string + ", Batch Size: " + str(batch)
            losses_figure_ax.plot(range(0, len(losses)), losses, label=label)
            scores_figure_ax.plot(range(0, len(scores)), scores, label=label)

    losses_figure_ax.set_title("Losses vs. Epochs (" + model_type + "+" + optimizer_type + ")")
    losses_figure_ax.set_xlabel("Epochs")
    losses_figure_ax.set_ylabel("Losses")
    losses_figure_ax.legend()
    losses_figure.savefig(model_type + "_" + optimizer_type + "_Losses_" + str(epochs) + "e.png")

    scores_figure_ax.set_title("Scores vs. Epochs")
    scores_figure_ax.set_xlabel("Epochs")
    scores_figure_ax.set_ylabel("Losses")
    scores_figure_ax.legend()
    scores_figure.savefig(model_type + "_" + optimizer_type + "_Scores_" + str(epochs) + "e.png")


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


main()
