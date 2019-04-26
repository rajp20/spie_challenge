import torch


class BaselineConvNet(torch.nn.Module):
    def __init__(self):
        super(BaselineConvNet, self).__init__()
        self.convolution_layer_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu = torch.nn.ReLU()
        self.maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
        self.convolution_layer_2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.convolution_layer_3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.convolution_layer_4 = torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5)
        self.convolution_layer_5 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5)
        self.fully_connected_layer = torch.nn.Linear(in_features=9216, out_features=14)

    def forward(self, x):
        x = self.convolution_layer_1(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)
        x = self.convolution_layer_2(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)
        x = self.convolution_layer_3(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)
        x = self.convolution_layer_4(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)
        x = self.convolution_layer_5(x)
        x = self.relu(x)
        x = self.maxpooling_layer(x)

        # flattening the tensor so that it can serve as input to a linear layer
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        return x
