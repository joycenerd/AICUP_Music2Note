import torch.nn as nn
import torch.nn.functional as F
import torch


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
