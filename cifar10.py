import os
import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from datetime import datetime
from torchvision import datasets


class Config:
    MODEL_PATH = 'cifar10-classification.pth.tar'
    BATCH_SIZE = 100
    EPOCHS = 250
    ETA = 1e-3
    TINY = False
    NORMALIZE = True


def load_data(tiny=Config.TINY, normalize=Config.NORMALIZE):
    data_dir = './data/cifar10'

    train_set = datasets.CIFAR10(data_dir, train=True, download=True)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True)

    # Train set processing
    train_input = torch.from_numpy(train_set.train_data)
    train_input = train_input.transpose(3, 1).transpose(2, 3).float()    
    train_target = torch.LongTensor(train_set.train_labels)

    # Test set processing
    test_input = torch.from_numpy(test_set.test_data).float()
    test_input = test_input.transpose(3, 1).transpose(2, 3).float()
    test_target = torch.LongTensor(test_set.test_labels)

    # Using GPU if available
    if (torch.cuda.is_available()):
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()

    # Reduce Dataset
    if tiny:
        print('** Reduce the data-set to the tiny setup (1000 samples)')
        train_input = train_input.narrow(0, 0, 1000)
        train_target = train_target.narrow(0, 0, 1000)
        test_input = test_input.narrow(0, 0, 1000)
        test_target = test_target.narrow(0, 0, 1000)

    # Normalize Dataset
    if normalize:
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target
 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        return x


"""
input:      3x32x32

conv1-1:    16x32x32
relu
conv1-2:    16x32x32
relu
pool1-1:    16x16x16

conv2-1:    32x16x16
relu
conv2-2:    32x16x16
relu
pool2-1:    32x8x8

conv3-1:    64x8x8
relu
conv3-2:    64x8x8
relu
pool3-1:    64x4x4

input2:     1x1024
fc1:    1024->2048
fc2:    2048->512
fc3:    512->10
"""


def create_conv_blocks():
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


def create_classifier():
    return [Flatten(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 2048),
    nn.BatchNorm1d(2048),
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.Linear(512, 10)]


def create_model():
    m = create_conv_blocks()
    m.extend(create_classifier())
    model = nn.Sequential(*m)
    return model

  
def save_model(model, path):
    if not path == None:
        torch.save(model, path)

        
def load_model(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None

      
def train_model(model, train_input, train_target,
                batch_size=Config.BATCH_SIZE,
                epochs=Config.EPOCHS,
                eta=Config.ETA,
                model_path=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=eta)

    for e in range(epochs):
        start_time_per_epoch = datetime.now()
        nb_errors = 0
        sum_loss = 0
        
        for b in range(0, train_input.size(0), batch_size):
            inputs = train_input.narrow(0, b, batch_size)
            targets = train_target.narrow(0, b, batch_size)
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Training loss
            sum_loss += loss.data
            
            # Training accuracy
            _, predicted = torch.max(outputs.data, 1)
            nb_errors += targets.ne(predicted).sum()  

        accuracy = 100 - (nb_errors.float() * 100 / train_input.size(0))
        elapsed = datetime.now() - start_time_per_epoch
        print('[{:5d}] Accuracy: {:3.2f}% - Loss: {:6.2f} - {}'.format(e + 1, accuracy, sum_loss, elapsed))
    
    if model_path is not None:
        save_model(model, model_path)


def compute_nb_errors(model, inputs, targets):
    outputs = model(inputs)
    _, predicted_classes = torch.max(outputs.data, 1)
    nb_errors = targets.ne(predicted_classes).sum()
    return nb_errors


def main():
    if torch.cuda.is_available():
        print ("CUDA IS AVAILABLE. USING CUDA")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Loading Data
    train_input, train_target, test_input, test_target = load_data()
    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)
    print('Data Loaded')

    # Creating Model
    model = create_model()
    if torch.cuda.is_available():
        model = model.cuda()
    print('Model Created')

    start_time = datetime.now()
    train_model(model, train_input, train_target, model_path=Config.MODEL_PATH)
    print('Elapsed time:', datetime.now() - start_time)

    nb_errors = compute_nb_errors(model, test_input, test_target)
    print ('Test error: {:.02f}% {:d} / {:d}'.format(100*nb_errors.float() / test_input.size(0), nb_errors, test_input.size(0)))


if __name__ == '__main__':
    main()
