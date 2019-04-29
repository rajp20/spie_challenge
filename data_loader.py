from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import torch


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
            self.image_path = "./datasets/breastpathq-test/test_patches/"

        if split is "train" or split is "val":
            label_csv = pd.read_csv(self.label_path, skiprows=1)
            for index, row in label_csv.iterrows():
                key = str(int(row[0])) + "_" + str(int(row[1]))
                self.dataset.append({"image": key, "label": row[2], "slide": int(row[0]), "rid": int(row[1])})
        if split is "test":
            for file in os.listdir(self.image_path):
                file_name = os.fsdecode(file).replace(".tif", "")
                file_name_split = file_name.split('_')
                self.dataset.append({"image": file_name, "label": 0, "slide": int(file_name_split[0]),
                                     "rid": int(file_name_split[1])})

    def __getitem__(self, index):
        set_of_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
             ])
        indexed_label = self.dataset[index]['label']
        indexed_image = set_of_transforms(Image.open(self.image_path + self.dataset[index]['image'] + ".tif"))
        return indexed_image, torch.FloatTensor([indexed_label])

    def __len__(self):
        return len(self.dataset)
