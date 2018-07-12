import os
import torch
import logging
from torch import Tensor, nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from datetime import datetime
from torchvision import datasets
from config import config_v3 as Config
import model as mymodels


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
        logging.info('** Reduce the data-set to the tiny setup (1000 samples)')
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
    if type(eta) == tuple:
        eta_batch = epochs // len(eta)
    else:
        optimizer = optim.SGD(model.parameters(), lr=eta)

    for e in range(epochs):
        if (type(eta) == tuple) and (e % eta_batch == 0):
            _eta = min(e // eta_batch, len(eta) - 1)
            logging.info('Changing ETA[{}] = {}'.format(_eta, eta[_eta]))
            optimizer = optim.SGD(model.parameters(), lr=eta[_eta])

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
        logging.info('[{:5d}] Accuracy: {:3.2f}% - Loss: {:6.2f} - {}'.format(e + 1, accuracy, sum_loss, elapsed))
    
    if model_path is not None:
        save_model(model, model_path)


def compute_nb_errors(model, inputs, targets):
    nb_errors = 0
    batch_size = Config.BATCH_SIZE

    for b in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs.narrow(0, b, batch_size)
        batch_targets = targets.narrow(0, b, batch_size)
        
        outputs = model(batch_inputs)
        _, predicted_classes = torch.max(outputs.data, 1)
        nb_errors += batch_targets.ne(predicted_classes).sum()

    return nb_errors



def main():
    logging.basicConfig(filemode='w', filename=Config.LOG_PATH, format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        logging.info('CUDA IS AVAILABLE. USING CUDA')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Loading Data
    train_input, train_target, test_input, test_target = load_data()
    train_input, train_target = Variable(train_input), Variable(train_target)
    test_input, test_target = Variable(test_input), Variable(test_target)
    logging.info('Data Loaded')

    # Creating Model
    model = mymodels.create_model('v2')
    if torch.cuda.is_available():
        model = model.cuda()
    logging.info('Model Created')

    start_time = datetime.now()
    train_model(model, train_input, train_target, model_path=Config.MODEL_PATH)
    logging.info('Elapsed time: {}'.format(datetime.now() - start_time))

    nb_errors = compute_nb_errors(model, test_input, test_target)
    logging.info('Test error: {:.02f}% {:d} / {:d}'.format(100*nb_errors.float() / test_input.size(0), nb_errors, test_input.size(0)))


if __name__ == '__main__':
    main()
