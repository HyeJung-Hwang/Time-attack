import torch.nn as nn
import torch.nn.functional as F
class GlobalMaxPooling(nn.Module):
    def forward(self, x):
        kernel_size = x.size()[2:]
        return F.max_pool1d(x, kernel_size )
    
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.contiguous().view(batch_size, -1)    
    
class CNN(nn.Module): 
    def __init__(self, n_classes = 2):   
        super(CNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d( in_channels = 1, out_channels= 64, kernel_size= 2, stride= 1),
            nn.BatchNorm1d(num_features = 64 ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= 2, stride= 1),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d( in_channels = 64, out_channels= 64, kernel_size= 2, stride= 1),
            nn.BatchNorm1d(num_features = 64 ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= 2, stride= 1),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d( in_channels = 64, out_channels=16, kernel_size= 2, stride= 1),
            nn.BatchNorm1d(num_features = 16 ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size= 2, stride= 1),
        )
        self.gmp_flatten = nn.Sequential(
            GlobalMaxPooling(),
            Flatten(),
        )
        self.FC = nn.Linear(16, 1)
        self.BN = nn.BatchNorm1d(16)

    def forward(self, x):
        x = x.view(-1,1,1*10)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.gmp_flatten(x)
        x = self.BN(x)
        x = self.FC(x)
        out = F.sigmoid(x)
        
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
