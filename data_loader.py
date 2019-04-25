from torch.utils.data import Dataset
import pandas as pd


class BreastPathQDataSet(Dataset):
    def __init__(self, split = "train"):
        self.images = "TODO"
        if split is "train":
            # Set up training data
            self.dataset_path = "./datasets/breastpathq/datasets/train/"
            self.label_path = "./datasets/breaspathq/datasets/train_labels.csv"
        elif split is "val":
            self.dataset_path = "./datasets/breastpathq/datasets/validation/"
            self.label_path = "./datasets/breastpathq-test/val_labels.csv"
        elif split is "train":
            self.dataset_path = "./datasets/breastpathq-test/test_patches/"

        if split is "train" or "val":
            label_csv = pd.read_csv(self.label_path, skiprows=1)
            for line in label_csv:
                print(line)

    def __getitem__(self, item):
        return item

    def __len__(self):
        return len(self.images)
