import torch.nn as nn
import torch.nn.functional as F
import torch
from options import opt


class NeuralNetWithRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetWithRNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) 
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[1])
        self.rnn = nn.RNN(hidden_size[1], hidden_size[1], num_layers=5, bidirectional=True)
        self.fc4 = nn.Linear(hidden_size[1]*2, hidden_size[1])
        self.fc5 =  nn.Linear(hidden_size[1], hidden_size[2])
        self.fc6 =  nn.Linear(hidden_size[2], hidden_size[3])
        self.fc7 =  nn.Linear(hidden_size[3], 2)
        self.fc8 =  nn.Linear(hidden_size[3], 1)
        self.relu =  nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out, hidden = self.rnn(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out =  self.relu(out)
        out =  self.fc6(out)
        out =  self.relu(out)
        out1 = torch.sigmoid(self.fc7(out))
        out2 = self.fc8(out)
        return out1, out2


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) 
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[1])
        self.fc4 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc5 =  nn.Linear(hidden_size[2], hidden_size[3])
        self.fc6 =  nn.Linear(hidden_size[3], 2)
        self.fc7 =  nn.Linear(hidden_size[3], 1)
        self.relu =  nn.ReLU()
        
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out =  self.relu(out)
        out1 = torch.sigmoid(self.fc6(out))
        out2 = self.fc7(out)
        return out1, out2


class Rnn(nn.Module):
    def __init__(self, in_features, hidden_size=100):
        super(Rnn,self).__init__()

        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=5, bidirectional=True)
        self.fc2 = nn.Linear(hidden_size*2, 2)
        self.fc3 = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out, hidden = self.rnn(out)
        out1 = torch.sigmoid(self.fc2(out))
        out2 = self.fc3(out)
        return out1, out2
    

class BiRNN(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers):
        super(BiRNN,self).__init__()
        self.hidden_size =  hidden_size
        self.num_layers =  num_layers
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout =  nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size*2, 2)
        self.fc3 =  nn.Linear(hidden_size*2, 1)
        
    def forward(self, x):
        # set initial state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda(opt.cuda_devices) # for bidirectional
        c0 =  torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda(opt.cuda_devices)
        
        # forward propagate LSTM
        out, hidden =  self.lstm(x, (h0,c0))
        out = self.dropout(out)
        
        # decode the hidden state of the last time step
        out1 =  torch.sigmoid(self.fc2(out))
        out2 =  self.fc3(out)
        return out1, out2
