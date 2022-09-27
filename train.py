
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
from models import LSTM_FC,MLP_BN
from data import load_data

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss


def train(model, train_loader,DEVICE, optimizer):
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
        
def evaluate(model, test_loader,DEVICE):
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
        
def train_model(train_loader, val_loader,DEVICE, num_epochs = 30):
    model = LSTM_FC().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) 
    for epoch in range(1, num_epochs + 1):
        since = time.time()                                     
        train(model, train_loader,DEVICE, optimizer)                   
        train_loss, train_acc = evaluate(model, train_loader,DEVICE)   
        val_loss, val_acc = evaluate(model, val_loader,DEVICE)         
        
        if val_acc>best_acc:  
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since 
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}'.format(train_loss, train_acc))   
        print('val Loss: {:.4f}, Accuracy: {:.2f}'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 

    model.load_state_dict(best_model_wts)  
    return model
    
def main():        
  DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
  
  data_path = "/home/cse_urp_dl2/Documents/hhj/BP/data/"
  train_loader, valid_loader = load_data(data_path)
  

  model = train_model(train_loader, valid_loader,DEVICE)  
  torch.save(model.state_dict(),"/home/cse_urp_dl2/Documents/hhj/BP/saved_models/check_FC00.pt" )
  
if __name__=="__main__":
    main()
