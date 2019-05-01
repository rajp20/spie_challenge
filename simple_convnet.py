import torch


class SimpleConvNet(torch.nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.convolution_layer_1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4)
        self.relu = torch.nn.ReLU()
        self.maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
        self.convolution_layer_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.convolution_layer_3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.convolution_layer_4 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4)
        self.convolution_layer_5 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4)
        self.fully_connected_layer = torch.nn.Linear(in_features=2704, out_features=1)

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
