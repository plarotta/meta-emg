import torch.nn as nn
import torch

class BasicCNN(nn.Module):
    def __init__(self, fc_dim=32):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(256,fc_dim)
        self.fc2 = nn.Linear(fc_dim,3)
        self.pool = nn.MaxPool1d(kernel_size=4)
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

        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x
