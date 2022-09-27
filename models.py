import torch.nn as nn
import torch.nn.functional as F

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