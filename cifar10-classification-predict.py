import os
import torch
from torch import Tensor, nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from datetime import datetime
from torchvision import datasets


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


def load_model(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None


def compute_nb_errors(model, inputs, targets):
    nb_errors = 0
    batch_size = inputs.size(0) / 50

    for b in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs.narrow(0, b, batch_size)
        batch_targets = targets.narrow(0, b, batch_size)
        
        outputs = model(batch_inputs)
        _, predicted_classes = torch.max(outputs.data, 1)
        nb_errors += batch_targets.ne(predicted_classes).sum()

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
    model = load_model('trained-model/v1/' + Config.MODEL_PATH)
    if model is None:
        print('Failed to load model')
        exit()
    if torch.cuda.is_available():
        model = model.cuda()
    print('Model Created')

    nb_errors_train = compute_nb_errors(model, train_input, train_target)
    nb_errors_test = compute_nb_errors(model, test_input, test_target)
    print ('Train error: {:.02f}% {:d} / {:d}'.format(100*nb_errors_train.float() / train_input.size(0), nb_errors_train, train_input.size(0)))
    print ('Test error: {:.02f}% {:d} / {:d}'.format(100*nb_errors_test.float() / test_input.size(0), nb_errors_test, test_input.size(0)))


if __name__ == '__main__':
    main()
