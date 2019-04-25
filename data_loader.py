from torch.utils.data import Dataset
import pandas as pd


class BreastPathQDataSet(Dataset):
    def __init__(self, split = "train"):
        self.labels = {}
        if split is "train":
            # Set up training data
            self.dataset_path = "./datasets/breastpathq/datasets/train/"
            self.label_path = "./datasets/breastpathq/datasets/train_labels.csv"
        elif split is "val":
            self.dataset_path = "./datasets/breastpathq/datasets/validation/"
            self.label_path = "./datasets/breastpathq-test/val_labels.csv"
        elif split is "train":
            self.dataset_path = "./datasets/breastpathq-test/test_patches/"

        if split is "train" or "val":
            label_csv = pd.read_csv(self.label_path, skiprows=1)
            for index, row in label_csv.iterrows():
                key = str(int(row[0])) + "_" + str(int(row[1]))
                self.labels[key] = row[2]
            print(len(self.labels))

    def __getitem__(self, item):
        return item

    def __len__(self):
        return len(self.labels)
