import torch
from torch import nn
from collections import OrderedDict

cag = {
    'mini-imagenet': 49, 'cifar10': 5184, 'cifar100': 1, 'stl10': 7, 'mnist':1, 'fashion-mnist':1
}


class SimNet(nn.Module):
    def __init__(self, data_name, in_channels, num_classes):
        super(SimNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('c1', nn.Conv2d(3, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu1', nn.ReLU()),
                ('s1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('c2', nn.Conv2d(9, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu2', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('c3', nn.Conv2d(27, 81, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
                ('relu3', nn.ReLU())
            ])
        )
        self.classifier = nn.Sequential(
            OrderedDict([
                ('f4', nn.Linear(cag['cifar10'], num_classes))
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def simnet(data_name, in_channels, num_classes):
    return SimNet(data_name, in_channels, num_classes)


# if __name__ == '__main__':
#     from torchsummary import summary

#     model = simnet()
#     summary(model, (3, 224, 224))
#     print(model)
