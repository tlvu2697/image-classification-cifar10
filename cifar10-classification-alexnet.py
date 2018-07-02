import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models
import torchvision.transforms as transforms
from datetime import datetime
import os


def save_model(model, path):
    if not path == None:
        torch.save(model, path)


def load_model(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None


def train_model(model, train_loader, model_path=None, learning_rate=1e-3, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    train_size = len(train_loader.dataset)

    for e in range(epochs):
        nb_errors = 0
        
        start_time_per_epoch = datetime.now()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted_classes =  torch.max(outputs.data, 1)
            nb_errors += labels.ne(predicted_classes).sum()
        
        print('[{:5d}] accuracy: {:.2f}% - {}'.format(e + 1, 100 - (nb_errors * 100 / train_size), datetime.now() - start_time_per_epoch))
    
    torch.save(model, model_path)


def compute_nb_errors(model, data_loader):
    nb_errors = 0

    for i, data in enumerate(data_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward
        outputs = model(inputs)

        _, predicted_classes =  torch.max(outputs.data, 1)
        nb_errors += labels.ne(predicted_classes).sum()
    print(nb_errors)
    return nb_errors


def load_data(batch_size=100, shuffle=False, num_workers=0):
    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.ToTensor()])

    cifar_train_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    x = torch.from_numpy(cifar_train_set.train_data).byte().narrow(0, 0, 500)
    padding = Variable(torch.zeros(500, 224, 224, 3))
    padded_inp = torch.cat((x, padding), 1)
    print(padded_inp.shape)

    cifar_test_set = datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(cifar_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, test_loader

#model = models.alexnet(pretrained=False).cuda()
train_loader, test_loader = load_data(batch_size=4)
#train_model(model, train_loader, 'alexnet.pth.tar')

#train_error = compute_nb_errors(model, train_loader)
#train_size = len(train_loader.dataset)
#print('--TRAIN ERROR: {:.2f}% - {} / {}'.format(train_error * 100 / train_size, train_error, train_size))
