import sys
import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import torchvision
import torchvision.transforms as transforms


class LSTM_Net(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*32, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size*20, hidden_size),
        #     nn.ReLU(),
        #     # nn.Linear(80, 20),
        #     # nn.ReLU(), 
        #     nn.Dropout(p=0.1),
        #     nn.Linear(hidden_size, output_size))
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (batch, seq_len, input_size)
        #x = F.relu(x)   #非线性激活
        s, b, h = x.shape  # x is output, size (batch, seq_len, hidden_size)
        x = x.contiguous().view(s, b*h)
        # x = self.fc(x)
        # #x = F.relu(x)   #非线性激活
        # x = x.view(s, b, -1)  # 把形状改回来
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x