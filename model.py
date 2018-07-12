from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        return x


## Model v1
def create_conv_blocks_v1():
    return [nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)]


def create_classifier_v1():
    return [Flatten(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 2048),
    nn.BatchNorm1d(2048),
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.Linear(512, 10)]
###########


## Model v2
def create_conv_blocks_v2():
    return [nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)]

def create_classifier_v2():
    return [Flatten(),
    nn.Dropout(0.5),
    nn.Linear(1024, 2048),
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.Dropout(0.5),
    nn.Linear(512, 10)]
###########


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def create_model(type):
    if type == 'v1':
        m = create_conv_blocks_v1()
        m.extend(create_classifier_v1())
        model = nn.Sequential(*m)
        return model
    if type == 'v2':
        m = create_conv_blocks_v2()
        m.extend(create_classifier_v2())
        model = nn.Sequential(*m)
        model.apply(init_weights)
        return model
    return None
