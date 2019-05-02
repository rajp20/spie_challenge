import torch


class ImprovedConvNet(torch.nn.Module):
    def __init__(self):
        super(ImprovedConvNet, self).__init__()
        self.input = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        self.block_1_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn_1 = torch.nn.BatchNorm2d(32)
        self.block_1_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.block_2_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn_2 = torch.nn.BatchNorm2d(64)
        self.block_2_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.block_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn_3 = torch.nn.BatchNorm2d(128)
        self.block_3_2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)

        self.block_4_1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.bn_4 = torch.nn.BatchNorm2d(256)
        self.block_4_2 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)

        self.fc_1 = torch.nn.Linear(in_features=18432, out_features=1)

        self.relu = torch.nn.ReLU()
        self.maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.input(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)

        # flattening the tensor so that it can serve as input to a linear layer
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)

        return x
