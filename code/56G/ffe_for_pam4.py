import torch
import pandas
import numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
# import math

# Hyper Parameters
data_dir = r'.\data\201801\C-band DML\25G APD\obtb'
tx_data_dir = r'.\data\201801'
batch_size = 256
rop = -9
dropout_prob = 0
weight_decay = 0
learning_rate = 0.003
epoch_num = 100
input_window = 101
dataset_length = 5 * (100000 - input_window + 1)


# Prepare the datasets
class RxDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, 1:], self.data[idx, 0].astype(numpy.int64)


if input_window % 2 == 0:
    raise ValueError("input_window must be odd")

data = numpy.empty((dataset_length, 1 + input_window))
row = 0
for i in range(5):
    rx_file_name = str(rop) + 'dBm' + str(i) + '.csv'
    rx_file_name = os.path.join(data_dir, rx_file_name)
    tx_file_name = 'pam4_' + str(i) + '.csv'
    tx_file_name = os.path.join(tx_data_dir, tx_file_name)
    raw_rx_data = pandas.read_csv(rx_file_name, header=None).values
    raw_tx_data = pandas.read_csv(tx_file_name, header=None).values
    for idx in range(raw_rx_data.shape[0] - input_window + 1):
        data[row, 1:] = raw_rx_data[idx: idx + input_window].T
        data[row, 0] = raw_tx_data[idx + int((input_window - 1) / 2)]
        row = row + 1
numpy.random.shuffle(data)

trainset_index = int(data.shape[0] * 0.6)
cvset_index = int(data.shape[0] * 0.8)
train_dataloader = DataLoader(RxDataset(data[:trainset_index, :]),
                              batch_size=batch_size,
                              shuffle=True)
cv_dataloader = DataLoader(RxDataset(data[trainset_index:cvset_index, :]),
                           batch_size=batch_size,
                           shuffle=True)
test_dataloader = DataLoader(RxDataset(data[cvset_index:, :]),
                             batch_size=batch_size,
                             shuffle=True)


# Define the network structure
# TODO Change the network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_window, 1)
        self.fc1.double()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fc1(x)
        return x


net = Net()
net.cuda()
print(net)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)

# Training
print('Start Training....')
train_loss = []
train_ber = []
cv_loss = []
cv_ber = []
for epoch in range(epoch_num):  # loop over the dataset multiple times
    # train
    net.train()
    for data in train_dataloader:
        # get the inputs
        rx_window, tx_symbol = data
        rx_window = Variable(rx_window.cuda())
        tx_symbol = Variable(tx_symbol.double().cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        loss.backward()
        optimizer.step()

    net.eval()
    # calculate train loss and acc
    running_loss = 0.0
    error = 0
    total = 0
    for data in train_dataloader:
        # get the inputs
        rx_window, tx_symbol = data
        rx_window = Variable(rx_window.cuda())
        tx_symbol = Variable(tx_symbol.double().cuda())
        # forward + backward + optimize
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.data[0]
        # make decision and calc BER
        outputs.data = outputs.data.cpu().squeeze()
        predicted = torch.zeros(outputs.data.size()).double()
        predicted[outputs.data <= 0.5] = 0
        predicted[(outputs.data <= 1.5) & (outputs.data > 0.5)] = 1
        predicted[(outputs.data <= 2.5) & (outputs.data > 1.5)] = 2
        predicted[outputs.data > 2.5] = 3
        total += 2 * batch_size
        target = tx_symbol.data.cpu()
        # import pdb; pdb.set_trace()
        temp = predicted[predicted != target]
        target = target[predicted != target]
        temp = torch.abs(temp - target) % 2
        temp[temp == 0] = 2
        error += temp.sum()
    train_loss.append(running_loss / len(train_dataloader))
    train_ber.append(error / total)
    print('#%d epoch train loss: %e train BER: %e' %
          (epoch + 1, train_loss[-1], train_ber[-1]))

    # calculate cv loss and acc
    running_loss = 0.0
    error = 0
    total = 0
    for data in cv_dataloader:
        # get the inputs
        rx_window, tx_symbol = data
        rx_window = Variable(rx_window.cuda())
        tx_symbol = Variable(tx_symbol.double().cuda())
        # forward + backward + optimize
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.data[0]
        # make decision and calc BER
        outputs.data = outputs.data.cpu().squeeze()
        predicted = torch.zeros(outputs.data.size()).double()
        predicted[outputs.data <= 0.5] = 0
        predicted[(outputs.data <= 1.5) & (outputs.data > 0.5)] = 1
        predicted[(outputs.data <= 2.5) & (outputs.data > 1.5)] = 2
        predicted[outputs.data > 2.5] = 3
        total += 2 * batch_size
        target = tx_symbol.data.cpu()
        temp = predicted[predicted != target]
        target = target[predicted != target]
        temp = torch.abs(temp - target) % 2
        temp[temp == 0] = 2
        error += temp.sum()
    cv_loss.append(running_loss / len(cv_dataloader))
    cv_ber.append(error / total)
    print('#%d epoch cv    loss: %e cv    BER: %e' %
          (epoch + 1, cv_loss[-1], cv_ber[-1]))
    print()
print('Finished Training....')

# Testing
print('Start Testing....')
error = 0
total = 0
for data in test_dataloader:
    rx_window, tx_symbol = data
    rx_window = Variable(rx_window.cuda())
    tx_symbol = Variable(tx_symbol.double().cuda())
    outputs = net(rx_window)
    # make decision and calc BER
    outputs.data = outputs.data.cpu().squeeze()
    predicted = torch.zeros(outputs.data.size()).double()
    predicted[outputs.data <= 0.5] = 0
    predicted[(outputs.data <= 1.5) & (outputs.data > 0.5)] = 1
    predicted[(outputs.data <= 2.5) & (outputs.data > 1.5)] = 2
    predicted[outputs.data > 2.5] = 3
    total += 2 * batch_size
    target = tx_symbol.data.cpu()
    temp = predicted[predicted != target]
    target = target[predicted != target]
    temp = torch.abs(temp - target) % 2
    temp[temp == 0] = 2
    error += temp.sum()
print('The test BER is %e' % (error / total))
print('Finished Testing....')

# draw the loss and acc curve
plt.figure()
plt.plot(range(len(train_loss)), train_loss, 'r',
         range(len(cv_loss)), cv_loss, 'b')
plt.title("Loss Learning Curve")
plt.yscale('log')
plt.figure()
plt.plot(range(len(train_ber)), train_ber, 'r',
         range(len(cv_ber)), cv_ber, 'b')
plt.title("BER Learning Curve")
plt.yscale('log')
plt.show()
