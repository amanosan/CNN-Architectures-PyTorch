# Implementation of VGG16 network.

import torch
import torch.nn as nn

# number of output channels after each conv layer, M denotes MaxPooling layer
VGG_TYPES = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


# the general vgg_net class
class VGG_NET(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_NET, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(VGG_TYPES['VGG16'])
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flattening the output of conv layers:
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # we know integer value is convolution layer:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


model = VGG_NET(in_channels=3, num_classes=1000)

# creating a random image of 224 x 224 to test the model
x = torch.randn(1, 3, 224, 224)

print(model(x).shape)
print(model)
