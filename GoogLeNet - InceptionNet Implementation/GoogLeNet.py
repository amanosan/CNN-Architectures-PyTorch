# Implementation of InceptionNet/GoogLeNet

import torch
import torch.nn as nn


# GoogLeNet / InceptionNet:
class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = Conv_Block(in_channels, out_channels=64,
                                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = Conv_Block(in_channels=64, out_channels=192, kernel_size=3,
                                stride=1, padding=1)

        # Inception blocks: in_channels,in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_Block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_Block(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception_Block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_Block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_Block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_Block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_Block(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception_Block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_Block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        if self.aux_logits:
            self.aux1 = Inception_aux(512, num_classes)
            self.aux2 = Inception_aux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)

        # Auxillary Softmax Classifier 1:
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxillary Softmax Classifier 2:
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return (aux1, aux2, x)
        else:
            return x


# Inception Block:
class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_Block, self).__init__()

        # branch 1:
        self.branch1 = Conv_Block(in_channels, out_1x1, kernel_size=1)

        # branch 2:
        self.branch2 = nn.Sequential(
            Conv_Block(in_channels, red_3x3, kernel_size=1),
            Conv_Block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # branch 3:
        self.branch3 = nn.Sequential(
            Conv_Block(in_channels, red_5x5, kernel_size=1),
            Conv_Block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # branch 4 (maxpooling):
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv_Block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        # Dimension of batch ----> N x filters x height x width
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# Auxillary Softmax Classifier (present in the 4th Inception Block)
# Look at the diagramatic architecture for more details
class Inception_aux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Inception_aux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Conv_Block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Convolution Block
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


if __name__ == '__main__':
    # creating three images
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(in_channels=3, num_classes=1000, aux_logits=True)

    print(f"Auxillary Classifier 1 Output Shape: {model(x)[0].shape}")
    print(f"Auxillary Classifier 2 Output Shape: {model(x)[1].shape}")
    print(f"Model Output Shape: {model(x)[2].shape}")
    # print(model)
