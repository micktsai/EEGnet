import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import data.dataloader as DL


class EEGnet(nn.Module):
    def __init__(self):
        super(EEGnet, self).__init__()

        # L-1
        self.l1_conv = nn.Conv2d(1, 16,kernel_size=(1,51), stride=(1,1), padding=(0, 25), bias=False)
        self.l1_batchnorm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # L-2
        self.l2_padding = nn.ZeroPad2d((0, 0, 0, 1))
        self.l2_conv = nn.Conv2d(16,32,kernel_size=(2,1), stride=(1, 1),groups=16, bias=False)
        self.l2_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l2_pooling = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0)
        # L-3
        self.l3_conv = nn.Conv2d(32,32,kernel_size=(1,15), stride=(1, 1), padding=(0, 7), bias=False)
        self.l3_batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l3_pooling = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0)
        # L-4
        self.fc1 = nn.Linear(in_features=736*2, out_features=2, bias=True)

    def  forward(self, data):
        data = data.clone().detach()
        # print(data.shape)
        # L-1
        data = self.l1_conv(data)
        data = self.l1_batchnorm(data)
        # print(data.shape)
        # L-2
        data= self.l2_padding(data)
        data = self.l2_conv(data)
        data = self.l2_batchnorm(data)
        data = F.elu(data, alpha=1.0)
        data = self.l2_pooling(data)
        data = F.dropout(data, p=0.25)
        # L-3
        data = self.l3_conv(data)
        data = self.l3_batchnorm(data)
        data = F.elu(data, alpha=1.0)
        data = self.l3_pooling(data)
        data = F.dropout(data, p=0.25)
        # L-4
        data = data.view(-1,736*2)
        data = self.fc1(data)
        # BCELoss
        # data = F.softmax(data, dim=1)
        # data = torch.argmax(data, dim=1)
        data = data.view(-1,1)
        return data

def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 36
    
    predicted = []
    
    for i in range(len(X)//batch_size):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    
    # CrossEntropyLoss
    predicted = torch.reshape(torch.from_numpy(predicted),(-1,2)).float()
    predicted = torch.argmax(predicted, dim=1)

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results

def main():
    train_data, train_label, test_data, test_label = DL.read_bci_data()
    net = EEGnet().double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    # print(train_label.shape)
    # net(train_data)
    batch_size = 36

    for epoch in range(10):  # loop over the dataset multiple times
        print("Epoch ", epoch)
        
        running_loss = 0.0
        for i in range(int(len(train_data)/batch_size)-1):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = torch.from_numpy(train_data[s:e])
            # BCELoss
            # labels = torch.FloatTensor(np.array([train_label[s:e]]).T*1.0)
            
            # CrossEntropyLoss
            labels = torch.LongTensor(np.array([train_label[s:e]]).T*1.0)
            # wrap them in Variable
            
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            
            outputs = torch.reshape(outputs,(-1,2))
            labels = torch.reshape(labels,(-1,))
            
            # CrossEntropyLoss
            # labels = torch.tensor(labels, dtype=torch.long) 
            loss = criterion(outputs.float(), labels)
            
            # BCELoss
            # loss = criterion(outputs.float(), labels.float())
            
            loss = loss.requires_grad_()
            loss.reduction = 'mean'
            # print(loss)
            loss.backward()
            
            
            optimizer.step()
            running_loss += loss.item()
        # Validation accuracy
        params = ["acc", "auc", "fmeasure"]
        print (params)
        print ("Training Loss ", running_loss)
        print ("Train - ", evaluate(net, train_data, train_label, params))
        # print ("Validation - ", evaluate(net, X_val, y_val, params))
        print ("Test - ", evaluate(net, test_data, test_label, params))

if __name__ == '__main__':
    main()