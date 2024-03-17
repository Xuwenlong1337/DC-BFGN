import numpy as np
from data_process_25 import generateds
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import torch
import torch.nn as nn
from DGnet_SE import DC
import torch.nn.functional as F
import utils
from tensorboardX import SummaryWriter
from torchsummary import summary

path = './Mydata/sig_1080data/'
txt = './Mydata/position/position_1080.txt'
format = '.csv'
history = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

reshape_size = 20
sub_channel = 8
GAT_in = 64

X, labels = generateds(path, txt, format, history)
X = (torch.from_numpy(X)).permute(0, 2, 1)  # transpose(1, 2)

trainx, testx, trainlabel, testlabel = train_test_split(X, labels, test_size=0.2, random_state=20)

sig_train, sig_test =trainx, testx #X_train,X_test
lab_train, lab_test =trainlabel, testlabel #y_train,y_test

lab_train = torch.from_numpy(lab_train)
lab_test = torch.from_numpy(lab_test)

batch_size = 64
train_tensor = data_utils.TensorDataset(sig_train, lab_train)
train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

batch_size = 64
test_tensor = data_utils.TensorDataset(sig_test, lab_test)
test_loader = data_utils.DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False,drop_last = True)

print(sig_train.size())
print(sig_test.size())

model = DC(reshape_size, sub_channel, GAT_in).double().to(device)#

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 300
print_every = 1
train_total_step = len(train_loader)
test_total_step = len(test_loader)

train_loss_list = []
train_acc_list = []
writer = SummaryWriter(logdir='K=12')

for epoch in range(num_epochs):
    train_loss = []
    train_acc = []

    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        train_signals = signals.to(device)
        train_labels = labels.to(device)
        train_outputs = model(train_signals.double())

        loss = criterion(train_outputs, train_labels.long())

        train_loss.append(loss.item())
        train_loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        loss.backward()
        optimizer.step()

        train_total = train_labels.size(0)
        _, train_predicted = torch.max(train_outputs.data, 1)

        correct = (train_predicted == train_labels.long()).sum().item()

        train_acc_list.append(correct / train_total)
        train_acc.append(correct / train_total)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, train_total_step, loss.item(),
                          (correct / train_total) * 100))
    train_loss_mean = np.mean(train_loss)
    train_acc_mean = np.mean(train_acc)

    writer.add_scalar('train accuracy', train_acc_mean, epoch)
    writer.add_scalar('train loss', train_loss_mean, epoch)
#  *********************************************************************************************************
    test_loss = []
    test_acc = []
    if (epoch + 1) >0:

        for i, (signals, labels) in enumerate(test_loader):
        # Run the forward pass
            test_signals = signals.to(device)
            test_labels = labels.to(device)

            test_outputs = model(test_signals.double())
            loss = criterion(test_outputs, test_labels.long())
            test_loss.append(loss.item())

            test_total = test_labels.size(0)
            _, test_predicted = torch.max(test_outputs.data, 1)
            correct = (test_predicted == test_labels.long()).sum().item()
            test_acc.append(correct / test_total)

        test_acc_mean = np.mean(test_acc)
        test_loss_mean = np.mean(test_loss)
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print('Average test loss:{:.4f}, Average test accuracy:{:.2f}%'.format(test_loss_mean, test_acc_mean*100))
        writer.add_scalars('accuracy', {'test': test_acc_mean, 'train_1': train_acc_mean}, epoch)
        writer.add_scalars('loss', {'test': test_loss_mean, 'train_1': train_loss_mean}, epoch)

writer.close()

