
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score,precision_score,recall_score
from torch.autograd import Variable
import time
import copy
import torch
from torch import Tensor
import glob
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
else:
    DEVICE = torch.device('cpu')
print(DEVICE)

train_x_valid = np.load("/home/cse_urp_dl2/Documents/hhj/BP/data/train_x.npy")
train_y_valid = np.load("/home/cse_urp_dl2/Documents/hhj/BP/data/train_y.npy").reshape(-1,1)
test_x = np.load("/home/cse_urp_dl2/Documents/hhj/BP/data/test_x.npy" )
test_y = np.load("/home/cse_urp_dl2/Documents/hhj/BP/data/test_y.npy" ).reshape(-1,1)

train_x, valid_x,train_y, valid_y =  train_test_split( train_x_valid, train_y_valid, test_size=0.1, random_state=42)

class BPDataset(Dataset): 
    def __init__(self,bp_data , label_data):
        self.x_data = bp_data
        self.y_data = label_data
    def __len__(self): 
        return len(self.x_data)
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])
        return x, y
        
train_dataset = BPDataset( train_x , train_y )
valid_dataset = BPDataset( valid_x , valid_y )
test_dataset = BPDataset(test_x , test_y )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size= 256, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=256, shuffle=True, num_workers=0)
class LSTM_vanilla_FC(nn.Module):
    def __init__(self, n_classes = 2):   
        super(LSTM_vanilla_FC, self).__init__()
        self.LSTM = nn.LSTM(input_size=1, hidden_size=16 ,batch_first=True)
        self.BN = nn.BatchNorm1d(16)
        self.FC_1 = nn.Linear(16, 1)
        self.FC_2 = nn.Linear(2, 1)


    def forward(self, x):

        out1, (hn, cn) = self.LSTM(x)

        final_state = out1[:,-1,:]
        out = self.BN(final_state)
        out = self.FC_1(out)
        out = F.sigmoid(out)
        return out
class LSTM_FC(nn.Module):
    def __init__(self, n_classes = 2):   
        super(LSTM_FC, self).__init__()
        self.LSTM = nn.LSTM(input_size=1, hidden_size=16 ,batch_first=True)
        self.BN = nn.BatchNorm1d(16)
        self.FC_1 = nn.Linear(16, 2)
        self.FC_2 = nn.Linear(2, 1)


    def forward(self, x):

        out1, (hn, cn) = self.LSTM(x)

        final_state = out1[:,-1,:]
        out = self.BN(final_state)
        out = self.FC_1(out)
        out = self.FC_2(out)
        out = F.sigmoid(out)
        return out
class MLP(nn.Module): 
    def __init__(self, n_classes = 2):   
        super(MLP, self).__init__()
        
#         self.BN = nn.BatchNorm1d(20)
        self.FC_1 = nn.Linear(10, 10)
        self.FC_2 = nn.Linear(10, 20)
        self.FC_3 = nn.Linear(20, 20)
        self.FC_4 = nn.Linear(20, 20)
        self.FC_5 = nn.Linear(20, 10)
        self.FC_6 = nn.Linear(10, 10)
        self.FC_7 = nn.Linear(10, 10)
        self.FC_8 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.view(-1, 1*10)
        out = self.FC_1(x)
        out = F.relu(out)
        
        out = self.FC_2(out)
        out = F.relu(out)
        
        out = self.FC_3(out)
        out = F.relu(out)
        
        out = self.FC_4(out)
        out = F.relu(out)
        
        out = self.FC_5(out)
        out = F.relu(out)
        
        out = self.FC_6(out)
        out = F.relu(out)
        
        out = self.FC_7(out)
        out = F.relu(out)
        
        out = self.FC_8(out)
        out = F.sigmoid(out)
        
        return out
        
class MLP_BN(nn.Module): 
    def __init__(self, n_classes = 2):   
        super(MLP_BN, self).__init__()
        
        self.FC_1 = nn.Linear(10, 10)
        self.FC_2 = nn.Linear(10, 20)
        self.FC_3 = nn.Linear(20, 20)
        self.FC_4 = nn.Linear(20, 20)
        self.FC_5 = nn.Linear(20, 10)
        self.FC_6 = nn.Linear(10, 10)
        self.FC_7 = nn.Linear(10, 10)
        self.FC_8 = nn.Linear(10, 1)
        
        self.BC_10 = nn.BatchNorm1d(10)
        self.BC_20 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = x.view(-1, 1*10)
        out = self.FC_1(x)
        out = F.relu(out)
        
        out = self.FC_2(out)
        out = self.BC_20(out)
        out = F.relu(out)
        
        out = self.FC_3(out)
        out = self.BC_20(out)
        out = F.relu(out)
        
        out = self.FC_4(out)
        out = self.BC_20(out)
        out = F.relu(out)
        
        out = self.FC_5(out)
        out = self.BC_10(out)
        out = F.relu(out)
        
        out = self.FC_6(out)
        out = self.BC_10(out)
        out = F.relu(out)
        
        out = self.FC_7(out)
        out = self.BC_10(out)
        out = F.relu(out)
        
        out = self.FC_8(out)
        out = F.sigmoid(out)
        
        return out

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss
LSTM_FC = LSTM_vanilla_FC().to(DEVICE)
optimizer = optim.Adam(LSTM_FC.parameters(), lr=0.001)

def train(model, train_loader, optimizer):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = BCELoss_class_weighted(torch.Tensor([1, 20]))
    model.train()                         
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)  
        optimizer.zero_grad()                              # optimizer gradient 
        output = model(data)                              
        loss =  criterion(output.to(torch.float32), target.to(torch.float32))                  
        loss.backward()                                   
        optimizer.step() 
        
def evaluate(model, test_loader):
    criterion = BCELoss_class_weighted(torch.Tensor([1, 20]))
    model.eval()      
    test_loss = 0     
    correct = 0       
    
    with torch.no_grad(): 
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)     
            output = model(data)                                  
            test_loss += float(criterion(output.to(torch.float32), target.to(torch.float32)))      
            pred = torch.round(output)
            
            correct += pred.eq(target.view_as(pred)).sum().item() 
    test_loss /= len(test_loader.dataset)                         
    test_accuracy = correct / len(test_loader.dataset)     
    return test_loss, test_accuracy
        
def train_model(model ,train_loader, val_loader, optimizer, num_epochs = 30):
    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) 
    for epoch in range(1, num_epochs + 1):
        since = time.time()                                     
        train(model, train_loader, optimizer)                   
        train_loss, train_acc = evaluate(model, train_loader)   
        val_loss, val_acc = evaluate(model, val_loader)         
        
        if val_acc>best_acc:  # update best accuracy
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))   
        print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

    model.load_state_dict(best_model_wts)  
    return model

model = train_model(LSTM_FC ,train_loader, valid_loader, optimizer)  
torch.save(model.state_dict(),"/home/cse_urp_dl2/Documents/hhj/BP/saved_models/LSTM_vanilla_FC00.pt" )