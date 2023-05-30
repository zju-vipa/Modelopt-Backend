import torch.nn as nn


cag = {
    'mini-imagenet': 49, 'cifar10': 1, 'cifar100': 1, 'stl10': 7, 'mnist':1, 'fashion-mnist':1
}


# no LRN
class AlexNet(nn.Module):
    def __init__(self, data_name, in_channels=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.data_name = data_name
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * cag[self.data_name] * 2 * cag[self.data_name], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * cag[self.data_name] * 2 * cag[self.data_name] )
        # x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


def alexnet(data_name, in_channels=3, num_classes=10):
    return AlexNet(data_name, in_channels=in_channels, num_classes=num_classes)
