import torch
from torch.utils.data import Dataset


class BreastPathQDataSet(Dataset):
    def __init__(self, split = "train"):
        self.images = "TODO"
        if split is "train":
            # Set up training data
            self.dataset_path = ""
        elif split is "val":
            self.dataset_path = ""
        elif split is "train":
            self.dataset_path = ""

    def __getitem__(self, item):
        return item

    def __len__(self):
        return len(self.images)
