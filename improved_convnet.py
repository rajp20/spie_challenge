import torch


class ImprovedConvNet(torch.nn.Module):
    def __init__(self):
        super(ImprovedConvNet, self).__init__()
        self.input = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)

        self.block_1_1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1)
        self.block_1_2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.block_1_3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.block_2_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.block_2_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.block_2_3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.block_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.block_3_2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.block_3_3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)

        self.block_4_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.block_4_2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.block_4_3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)

        self.block_5_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.block_5_2 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.block_5_3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)

        self.fc_1 = torch.nn.Linear(in_features=1152, out_features=512)
        self.fc_2 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc_3 = torch.nn.Linear(in_features=256, out_features=1)

        self.relu = torch.nn.ReLU()
        self.maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.input(x)

        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.block_1_3(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.block_2_3(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_3_3(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.block_4_3(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.block_5_3(x)

        x = self.relu(x)
        x = self.maxpooling_layer(x)

        # flattening the tensor so that it can serve as input to a linear layer
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        return x
