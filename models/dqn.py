import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, in_size: torch.Size, out_size: torch.Size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_size[0], 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, out_size[0])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def dqn(pretrained=False, model_weights=None, **kwargs):
    model = DQN(**kwargs)
    if pretrained:
        if model_weights is None:
            raise AssertionError("No default weights to load")
        else:
            model.load_state_dict(model_weights)

    return model
