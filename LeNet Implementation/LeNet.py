# Note - LeNet uses tanh, but we will use ReLu.

import torch
import torch.nn as nn


# creating the class
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5))
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # dimension --> num_examples x 120 x 1 x 1
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # dimension --> num_examples x 120
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x


# dimension --> (num_examples, channels, height, width)
x = torch.randn(64, 1, 32, 32)
model = LeNet()
# the output should be (64, 10) --> 64 samples and 10 classes
print(model(x).shape)
