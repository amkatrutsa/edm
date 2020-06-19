import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, input_features):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 100)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x