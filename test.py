from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
from data import load_data,load_test_data

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss

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
def prediction(model, data_loader,DEVICE):
    model.eval()
    predlist=torch.zeros(0,dtype=torch.float32, device=DEVICE)
    lbllist=torch.zeros(0,dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(DEVICE)        
            label = label.to(DEVICE)     
            outputs = model(data)         
            pred = torch.round(outputs)

            predlist=torch.cat([predlist,pred.view(-1)])
            lbllist=torch.cat([lbllist,label.view(-1)])

    print(classification_report(lbllist.cpu().numpy(), predlist.cpu().numpy())) 
    print("Precision  :\t"+str(precision_score(lbllist.cpu().numpy(), predlist.cpu().numpy())))
    print("Recall     :\t"+str(recall_score(lbllist.cpu().numpy(), predlist.cpu().numpy() )))
    print("F1-score     :\t"+str(f1_score(lbllist.cpu().numpy(), predlist.cpu().numpy() )))
    precision, recall, thresholds = precision_recall_curve(lbllist.cpu().numpy(), predlist.cpu().numpy())
    auc_precision_recall = auc(recall, precision)
    print("AUPRC     :\t" + str(auc_precision_recall))
    print("AUROC     :\t" + str(roc_auc_score(lbllist.cpu().numpy(), predlist.cpu().numpy())))
    return   
    
def main():        
  DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
  model_path  = "/home/cse_urp_dl2/Documents/hhj/BP/saved_models/MLP_BN00.pt"
  data_path = "/home/cse_urp_dl2/Documents/hhj/BP/data/"
  
  model = MLP_BN().to(DEVICE)
  model.load_state_dict(torch.load(model_path))
  
  test_loader = load_test_data(data_path)
  prediction(model, test_loader,DEVICE)
  
if __name__=="__main__":
    main()