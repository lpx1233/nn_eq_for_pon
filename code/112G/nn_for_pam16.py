import torch
import pandas
import numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os


print('Starting time is ' + time.strftime('%Y-%m-%d %H:%M:%S'))
# Hyper Parameters
data_dir = r'.\data\201805\20180519\pam16-soa-btb'
tx_data_dir = r'.\data\201805'
batch_size = 512
dropout_prob = 0.1
weight_decay = 0.003
learning_rate = 0.0003
input_window = 101
epoch_num = 300
rop = -7
data_type = 'prbs15'
if(data_type == 'random'):
    dataset_length = 5 * (100000 - input_window + 1)
    data_dir = os.path.join(data_dir, 'random')
elif(data_type == 'prbs15'):
    dataset_length = 10 * (32767 - input_window + 1)
    data_dir = os.path.join(data_dir, 'prbs15')


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
if(data_type == 'random'):
    for i in range(5):
        rx_file_name = str(rop) + 'dBm' + str(i) + '.csv'
        rx_file_name = os.path.join(data_dir, rx_file_name)
        tx_file_name = 'pam16_' + str(i) + '.csv'
        tx_file_name = os.path.join(tx_data_dir, tx_file_name)
        raw_rx_data = pandas.read_csv(rx_file_name, header=None).values
        raw_tx_data = pandas.read_csv(tx_file_name, header=None).values
        for idx in range(raw_rx_data.shape[0] - input_window + 1):
            data[row, 1:] = raw_rx_data[idx: idx + input_window].T
            data[row, 0] = raw_tx_data[idx + int((input_window - 1) / 2)]
            row = row + 1
elif(data_type == 'prbs15'):
    for i in range(10):
        rx_file_name = str(rop) + 'dBm' + str(i) + '.csv'
        rx_file_name = os.path.join(data_dir, rx_file_name)
        tx_file_name = 'prbs15_pam16.csv'
        tx_file_name = os.path.join(tx_data_dir, tx_file_name)
        raw_rx_data = pandas.read_csv(rx_file_name, header=None).values
        raw_tx_data = pandas.read_csv(tx_file_name, header=None).values
        for idx in range(raw_rx_data.shape[0] - input_window + 1):
            data[row, 1:] = raw_rx_data[idx: idx + input_window].T
            data[row, 0] = raw_tx_data[idx + int((input_window - 1) / 2)]
            row = row + 1
# numpy.random.shuffle(data)

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
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv1.double()
        self.bn1 = nn.BatchNorm1d(6)
        self.bn1.double()

        self.conv2 = nn.Conv1d(6, 16, 5)
        self.conv2.double()
        self.bn2 = nn.BatchNorm1d(16)
        self.bn2.double()

        self.fc1 = nn.Linear(16 * (input_window - 8), 600)
        self.fc1.double()
        self.bn3 = nn.BatchNorm1d(600)
        self.bn3.double()

        self.fc2 = nn.Linear(600, 120)
        self.fc2.double()
        self.bn4 = nn.BatchNorm1d(120)
        self.bn4.double()

        self.fc3 = nn.Linear(120, 84)
        self.fc3.double()
        self.bn5 = nn.BatchNorm1d(84)
        self.bn5.double()

        self.fc4 = nn.Linear(84, 16)
        self.fc4.double()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        # weight init: Kaiming He init
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc4(x)
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
optimizer = optim.Adam(net.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=0.33,
                                                 patience=5,
                                                 verbose=True)

def cal_pam16_biterror(predicted, target):
    """Calculate the bit error of PAM8 using Grey coding

    Args:
        predicted (torch.Tensor): The predicted PAM8 sequence
        target (torch.Tensor): The target PAM8 sequence

    Returns:
        int: The bit error between the predicted and target PAM8 sequence
    """
    temp = torch.abs(predicted - target)
    temp[temp == 0] = 0
    temp[temp == 1] = 1
    temp[temp == 2] = 2
    # delta = 3
    temp[(predicted == 7) & (target == 4)] = 1
    temp[(predicted == 4) & (target == 7)] = 1
    temp[(predicted == 6) & (target == 3)] = 3
    temp[(predicted == 3) & (target == 6)] = 3
    temp[(predicted == 5) & (target == 2)] = 1
    temp[(predicted == 2) & (target == 5)] = 1
    temp[(predicted == 4) & (target == 1)] = 3
    temp[(predicted == 1) & (target == 4)] = 3
    temp[(predicted == 3) & (target == 0)] = 1
    temp[(predicted == 0) & (target == 3)] = 1
    temp[(predicted == 5) & (target == 8)] = 3
    temp[(predicted == 8) & (target == 5)] = 3
    temp[(predicted == 6) & (target == 9)] = 1
    temp[(predicted == 9) & (target == 6)] = 1
    temp[(predicted == 7) & (target == 10)] = 3
    temp[(predicted == 10) & (target == 7)] = 3
    temp[(predicted == 8) & (target == 11)] = 1
    temp[(predicted == 11) & (target == 8)] = 1
    temp[(predicted == 9) & (target == 12)] = 3
    temp[(predicted == 12) & (target == 9)] = 3
    temp[(predicted == 10) & (target == 13)] = 1
    temp[(predicted == 13) & (target == 10)] = 1
    temp[(predicted == 11) & (target == 14)] = 3
    temp[(predicted == 14) & (target == 11)] = 3
    temp[(predicted == 12) & (target == 15)] = 1
    temp[(predicted == 15) & (target == 12)] = 1
    # delta = 4
    temp[temp == 4] = 2
    # delta = 5
    temp[(predicted == 0) & (target == 5)] = 3
    temp[(predicted == 5) & (target == 0)] = 3
    temp[(predicted == 1) & (target == 6)] = 1
    temp[(predicted == 6) & (target == 1)] = 1
    temp[(predicted == 2) & (target == 7)] = 3
    temp[(predicted == 7) & (target == 2)] = 3
    temp[(predicted == 3) & (target == 8)] = 3
    temp[(predicted == 8) & (target == 3)] = 3
    temp[(predicted == 4) & (target == 9)] = 3
    temp[(predicted == 9) & (target == 4)] = 3
    temp[(predicted == 5) & (target == 10)] = 1
    temp[(predicted == 10) & (target == 5)] = 1
    temp[(predicted == 6) & (target == 11)] = 3
    temp[(predicted == 11) & (target == 6)] = 3
    temp[(predicted == 7) & (target == 12)] = 3
    temp[(predicted == 12) & (target == 7)] = 3
    temp[(predicted == 8) & (target == 13)] = 3
    temp[(predicted == 13) & (target == 8)] = 3
    temp[(predicted == 9) & (target == 14)] = 1
    temp[(predicted == 14) & (target == 9)] = 1
    temp[(predicted == 10) & (target == 15)] = 3
    temp[(predicted == 15) & (target == 10)] = 3
    # delta = 6
    temp[(predicted == 0) & (target == 6)] = 2
    temp[(predicted == 6) & (target == 0)] = 2
    temp[(predicted == 1) & (target == 7)] = 2
    temp[(predicted == 7) & (target == 1)] = 2
    temp[(predicted == 2) & (target == 8)] = 4
    temp[(predicted == 8) & (target == 2)] = 4
    temp[(predicted == 3) & (target == 9)] = 4
    temp[(predicted == 9) & (target == 3)] = 4
    temp[(predicted == 4) & (target == 10)] = 2
    temp[(predicted == 10) & (target == 4)] = 2
    temp[(predicted == 5) & (target == 11)] = 2
    temp[(predicted == 11) & (target == 5)] = 2
    temp[(predicted == 6) & (target == 12)] = 4
    temp[(predicted == 12) & (target == 6)] = 4
    temp[(predicted == 7) & (target == 13)] = 4
    temp[(predicted == 13) & (target == 7)] = 4
    temp[(predicted == 8) & (target == 14)] = 2
    temp[(predicted == 14) & (target == 8)] = 2
    temp[(predicted == 9) & (target == 15)] = 2
    temp[(predicted == 15) & (target == 9)] = 2
    # delta = 7
    temp[(predicted == 0) & (target == 7)] = 1
    temp[(predicted == 7) & (target == 0)] = 1
    temp[(predicted == 1) & (target == 8)] = 3
    temp[(predicted == 8) & (target == 1)] = 3
    temp[(predicted == 2) & (target == 9)] = 3
    temp[(predicted == 9) & (target == 2)] = 3
    temp[(predicted == 3) & (target == 10)] = 3
    temp[(predicted == 10) & (target == 3)] = 3
    temp[(predicted == 4) & (target == 11)] = 1
    temp[(predicted == 11) & (target == 4)] = 1
    temp[(predicted == 5) & (target == 12)] = 3
    temp[(predicted == 12) & (target == 5)] = 3
    temp[(predicted == 6) & (target == 13)] = 3
    temp[(predicted == 13) & (target == 6)] = 3
    temp[(predicted == 7) & (target == 14)] = 3
    temp[(predicted == 14) & (target == 7)] = 3
    temp[(predicted == 8) & (target == 15)] = 1
    temp[(predicted == 15) & (target == 8)] = 1
    # delta = 8
    temp[temp == 8] = 2
    # delta = 9
    temp[(predicted == 0) & (target == 9)] = 3
    temp[(predicted == 9) & (target == 0)] = 3
    temp[(predicted == 1) & (target == 10)] = 3
    temp[(predicted == 10) & (target == 1)] = 3
    temp[(predicted == 2) & (target == 11)] = 3
    temp[(predicted == 11) & (target == 2)] = 3
    temp[(predicted == 3) & (target == 12)] = 1
    temp[(predicted == 12) & (target == 3)] = 1
    temp[(predicted == 4) & (target == 13)] = 3
    temp[(predicted == 13) & (target == 4)] = 3
    temp[(predicted == 5) & (target == 14)] = 3
    temp[(predicted == 14) & (target == 5)] = 3
    temp[(predicted == 6) & (target == 15)] = 3
    temp[(predicted == 15) & (target == 6)] = 3
    # delta = 10
    temp[(predicted == 0) & (target == 10)] = 4
    temp[(predicted == 10) & (target == 0)] = 4
    temp[(predicted == 1) & (target == 11)] = 4
    temp[(predicted == 11) & (target == 1)] = 4
    temp[(predicted == 2) & (target == 12)] = 2
    temp[(predicted == 12) & (target == 2)] = 2
    temp[(predicted == 3) & (target == 13)] = 2
    temp[(predicted == 13) & (target == 3)] = 2
    temp[(predicted == 4) & (target == 14)] = 4
    temp[(predicted == 14) & (target == 4)] = 4
    temp[(predicted == 5) & (target == 15)] = 4
    temp[(predicted == 15) & (target == 5)] = 4
    # delta = 11
    temp[(predicted == 0) & (target == 11)] = 3
    temp[(predicted == 11) & (target == 0)] = 3
    temp[(predicted == 1) & (target == 12)] = 3
    temp[(predicted == 12) & (target == 1)] = 3
    temp[(predicted == 2) & (target == 13)] = 1
    temp[(predicted == 13) & (target == 2)] = 1
    temp[(predicted == 3) & (target == 14)] = 3
    temp[(predicted == 14) & (target == 3)] = 3
    temp[(predicted == 4) & (target == 15)] = 3
    temp[(predicted == 15) & (target == 4)] = 3
    # delta = 12
    temp[temp == 12] = 2
    # delta = 13
    temp[(predicted == 0) & (target == 13)] = 3
    temp[(predicted == 13) & (target == 0)] = 3
    temp[(predicted == 1) & (target == 14)] = 1
    temp[(predicted == 14) & (target == 1)] = 1
    temp[(predicted == 2) & (target == 15)] = 3
    temp[(predicted == 15) & (target == 2)] = 3
    # delta = 14
    temp[temp == 14] = 2
    # delta = 15
    temp[temp == 15] = 1
    return temp.sum().item()

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
        # forward
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += 4 * batch_size
        error += cal_pam16_biterror(predicted, tx_symbol.data)
    # import pdb; pdb.set_trace()
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
        # forward
        outputs = net(rx_window)
        loss = criterion(outputs, tx_symbol)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += 4 * batch_size
        error += cal_pam16_biterror(predicted, tx_symbol.data)
    cv_loss.append(running_loss / len(cv_dataloader))
    cv_ber.append(error / total)
    print('#%d epoch cv    loss: %e cv    BER: %e' %
          (epoch + 1, cv_loss[-1], cv_ber[-1]))
    print()
    # get the next lr using scheduler
    scheduler.step(cv_ber[-1])

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
    total += 4 * batch_size
    error += cal_pam16_biterror(predicted, tx_symbol.data)
test_ber = error / total
print('The test BER is %e' % test_ber)
print('Finished Testing....')
print('Ending time is ' + time.strftime('%Y-%m-%d %H:%M:%S'))
# draw the loss and acc curve
plt.figure()
plt.plot(range(len(train_loss)), train_loss, 'r',
         range(len(cv_loss)), cv_loss, 'b')
plt.title("Loss Learning Curve")
plt.yscale('log')
plt.savefig(str(rop) + '_loss.jpg')
plt.figure()
plt.plot(range(len(train_ber)), train_ber, 'r',
         range(len(cv_ber)), cv_ber, 'b')
plt.title("BER Learning Curve")
plt.yscale('log')
plt.savefig(str(rop) + '_BER.jpg')
plt.show()
