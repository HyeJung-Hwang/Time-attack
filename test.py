from torch.utils.data import DataLoader
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score,precision_score,recall_score
from torch.autograd import Variable
import time
import copy
import torch
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
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
from sklearn.metrics import auc, plot_precision_recall_curve, precision_recall_curve
from models import CNN
from data import load_test_data


    
def prediction( data_loader,DEVICE):
    model = CNN().to(DEVICE)
    PATH = "/home/cse_urp_dl2/Documents/hhj/BP/saved_models/CNN00.pt"
    model.load_state_dict(torch.load(PATH))
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
        
    # Classification Report
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
  data_path = "/home/cse_urp_dl2/Documents/hhj/BP/data/"
  test_loader = load_test_data(data_path)
  
  
  prediction( test_loader,DEVICE)

if __name__=="__main__":
    main()
