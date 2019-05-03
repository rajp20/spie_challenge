from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import torch
import random


class BreastPathQDataSet(Dataset):
    def __init__(self, split="train"):
        self.dataset = []
        if split is "train":
            # Set up training data
            self.image_path = "./datasets/breastpathq/datasets/train/"
            self.label_path = "./datasets/breastpathq/datasets/train_labels.csv"
        elif split is "val":
            self.image_path = "./datasets/breastpathq/datasets/validation/"
            self.label_path = "./datasets/breastpathq-test/val_labels.csv"
        elif split is "test":
            self.image_path = "./datasets/breastpathq/datasets/test/"
            self.label_path = "./datasets/breastpathq/datasets/test_labels.csv"

        label_csv = pd.read_csv(self.label_path, skiprows=1)
        for index, row in label_csv.iterrows():
            key = str(int(row[0])) + "_" + str(int(row[1]))
            self.dataset.append({"image": key, "label": row[2], "slide": int(row[0]), "rid": int(row[1])})

    # apply transoforms to all tensors in list x
    def _iterate_transforms(self, transform, x):
        for i, xi in enumerate(x):
            x[i] = transform(x[i])
        return x

    # redefining composed transform so that it uses the _iterate_transforms function
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            for transform in self.transforms:
                x = self._iterate_transforms(transform, x)
            return x

    # class to rerandomize the vertical flip transformation
    class RandomVerticalFlipGenerator(object):
        def __call__(self, img):
            self.random_n = random.uniform(0, 1)
            return img

    # class to perform vertical flip using randomization provided by gen
    class RandomVerticalFlip(object):
        def __init__(self, gen):
            self.gen = gen

        def __call__(self, img):
            if self.gen.random_n < 0.5:
                return torch.flip(img, [2])
            return img

    # class to rerandomize the horizontal flip transformation
    class RandomHorizontalFlipGenerator(object):
        def __call__(self, img):
            self.random_n = random.uniform(0, 1)
            return img

    # class to perform horizontal flip using randomization provided by gen
    class RandomHorizontalFlip(object):
        def __init__(self, gen):
            self.gen = gen

        def __call__(self, img):
            if self.gen.random_n < 0.5:
                return torch.flip(img, [1])
            return img

    def __getitem__(self, index):
        indexed_label = self.dataset[index]['label']
        image = Image.open(self.image_path + self.dataset[index]['image'] + ".tif")

        set_of_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
             ])

        indexed_image = set_of_transforms(image)
        return indexed_image, torch.FloatTensor([indexed_label])

    def __len__(self):
        return len(self.dataset)
