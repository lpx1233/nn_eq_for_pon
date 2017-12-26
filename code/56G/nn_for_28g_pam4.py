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
data_dir = r'..\data\28g_pam4_rand'
up_sample_rate = 20
batch_size = 256
rop = -13
dropout_prob = 0
weight_decay = 0
learning_rate = 0.001
epoch_num = 100
input_window = 101


# Prepare the datasets
class RxDataset(Dataset):
    def __init__(self, input_data, target):
        self.input_data = input_data
        self.target = target

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx, :], self.target[idx, 0]


# TODO add assert
if input_window % 2 == 0:
    raise ValueError("input_window must be odd")

dataset_length = 5 * (100000 - 101 + 1)
input_data = numpy.empty((dataset_length, input_window))
target = numpy.empty((dataset_length, 1))
for i in range(5):
    rx_file_name = str(rop) + 'dBm' + str(i) + '.csv'
    rx_file_name = os.path.join(data_dir, rx_file_name)
    tx_file_name = 'pam4_' + str(i) + '.csv'
    tx_file_name = os.path.join(data_dir, tx_file_name)
    # TODO complete this
    raw_rx_data = pandas.read_csv(rx_file_name, header=None) \
                        .values[::up_sample_rate]
    raw_tx_data = pandas.read_csv(tx_file_name, header=None).values

    for idx in range(raw_rx_data.shape[0] - input_window + 1):
        input_data[row, :] = raw_rx_data[idx: idx + input_window].T
        target[row, :] = raw_rx_data[idx + int((input_window - 1) / 2)]
print(input_data.shape)
print(target.shape)

trainset_index = int(input_data.shape[0] * 0.6)
cvset_index = int(input_data.shape[0] * 0.8)

train_dataloader = DataLoader(RxDataset(input_data[:trainset_index, :],
                                        target[:trainset_index, :]),
                              batch_size=batch_size,
                              shuffle=True)
cv_dataloader = DataLoader(RxDataset(input_data[trainset_index:cvset_index, :],
                                     target[trainset_index:cvset_index, :]),
                           batch_size=batch_size,
                           shuffle=True)
test_dataloader = DataLoader(RxDataset(input_data[cvset_index:, :],
                                       target[cvset_index:, :]),
                             batch_size=batch_size,
                             shuffle=True)


# Define the network structure
# TODO Change the network structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 3)
        self.conv1.double()
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.conv2.double()
        # TODO change the input_channel of fc1 if window is changed
        self.fc1 = nn.Linear(16 * (input_window - 4), 120)
        self.fc1.double()
        self.fc2 = nn.Linear(120, 84)
        self.fc2.double()
        self.fc3 = nn.Linear(84, 4)
        self.fc3.double()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        # For SELU only
        # nn.init.normal(list(self.conv1.parameters())[0], std=math.sqrt(1/5))
        # nn.init.normal(list(self.conv2.parameters())[0], std=math.sqrt(1/5))
        # nn.init.normal(list(self.fc1.parameters())[0],
        #                std=math.sqrt(1/(16*(input_window - 4))))
        # nn.init.normal(list(self.fc2.parameters())[0], std=math.sqrt(1/120))
        # nn.init.normal(list(self.fc3.parameters())[0], std=math.sqrt(1/84))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net.cuda()
print(net)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.003)
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
        tx_symbol = Variable(tx_symbol.cuda())
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
        tx_symbol = Variable(tx_symbol.cuda())
        # forward + backward + optimize
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += 2 * batch_size
        target = tx_symbol.data
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
        tx_symbol = Variable(tx_symbol.cuda())
        # forward + backward + optimize
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += 2 * batch_size
        target = tx_symbol.data
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
    tx_symbol = Variable(tx_symbol.cuda())
    outputs = net(rx_window)
    _, predicted = torch.max(outputs.data, 1)
    total += 2 * batch_size
    target = tx_symbol.data
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
