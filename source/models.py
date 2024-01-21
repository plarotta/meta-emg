import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import weight_norm


class BasicCNN(nn.Module):
    def __init__(self, fc_dim=32, input_seq_len=25):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64*(input_seq_len-6),fc_dim)
        self.fc2 = nn.Linear(fc_dim,3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = torch.permute(x, (0,2,1))
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = torch.flatten(x,start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x



## nabbed from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class TCN(nn.Module):
    def __init__(self, in_channels, layer_channels, seq_len, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=in_channels, num_channels=layer_channels, kernel_size=kernel_size, dropout=dropout)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(25*seq_len, 64)
        self.linear2 = nn.Linear(64,3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.float()
        x = torch.permute(x, (0,2,1))
        x = self.tcn(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return(x)


if __name__ == '__main__':
    # Just for debugging purposes
    seq_len = 40
    net = TCN(8, 3*[25], seq_len)
    sam = torch.zeros((32,seq_len,8))

    sam = torch.permute(sam, (0,2,1))
    print(net(sam).shape)