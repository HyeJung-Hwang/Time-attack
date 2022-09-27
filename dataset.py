from torch.utils.data import Dataset
from torch import Tensor
import torch

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