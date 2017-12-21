import torch
import pandas
import numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import math

# Hyper Parameters
up_sample_rate = 1
batch_size = 256
rop = -13
dropout_prob = 0
weight_decay = 0
learning_rate = 0.001
epoch_num = 100
input_window = 101


# Prepare the datasets
class RxDataset(Dataset):
    def __init__(self, rx_data, tx_data, window=11):
        self.rx_data = rx_data
        self.tx_data = tx_data
        self.window = window
        if window % 2 is 0:
            self.window = 11
            print('WARNING: the window cannot be even. reset to 11.')

    def __len__(self):
        return self.tx_data.shape[0] - self.window + 1

    def __getitem__(self, idx):
        rx_window = self.rx_data[idx: idx + self.window, 0]
        offset = int((self.window - 1) / 2)
        tx_symbol = self.tx_data[idx + offset, 0]
        return rx_window, tx_symbol


for i in range(3):
    rx_file_name = '.\\data_28g_pam4_rand\\' + str(rop) + 'dBm' + str(i) + '.csv'
    tx_file_name = '.\\data_28g_pam4_rand\\pam4_' + str(i) + '.csv'
    if i == 0:
        rx_data = pandas.read_csv(rx_file_name, sep=',', header=None) \
                        .values[::up_sample_rate]
        tx_data = pandas.read_csv(tx_file_name, sep=',', header=None) \
                        .values
    else:
        rx_data = numpy.concatenate(
            (rx_data,
             pandas.read_csv(rx_file_name, sep=',', header=None)
                   .values[::up_sample_rate]),
            axis=0
        )
        tx_data = numpy.concatenate(
            (tx_data,
             pandas.read_csv(tx_file_name, sep=',', header=None)
                   .values),
            axis=0
        )
train_dataset = RxDataset(rx_data, tx_data, input_window)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

rx_file_name = '.\\data_28g_pam4_rand\\' + str(rop) + 'dBm3.csv'
tx_file_name = '.\\data_28g_pam4_rand\\pam4_3.csv'
rx_data = pandas.read_csv(rx_file_name, sep=',', header=None) \
                .values[::up_sample_rate]
tx_data = pandas.read_csv(tx_file_name, sep=',', header=None) \
                .values
cv_dataset = RxDataset(rx_data, tx_data, input_window)
cv_dataloader = DataLoader(cv_dataset,
                           batch_size=batch_size,
                           shuffle=True)

rx_file_name = '.\\data_28g_pam4_rand\\' + str(rop) + 'dBm4.csv'
tx_file_name = '.\\data_28g_pam4_rand\\pam4_4.csv'
rx_data = pandas.read_csv(rx_file_name, sep=',', header=None) \
                .values[::up_sample_rate]
tx_data = pandas.read_csv(tx_file_name, sep=',', header=None) \
                .values
test_dataset = RxDataset(rx_data, tx_data, input_window)
test_dataloader = DataLoader(test_dataset,
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
