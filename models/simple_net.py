import torch.nn as nn
import torch.nn.functional as F
import torch


class SimpleNet(nn.Module):
    def __init__(self, in_size: torch.Size, out_size: torch.Size):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_size[0], 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_size[0])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def simplenet(pretrained=False, model_weights=None, **kwargs):
    model = SimpleNet(**kwargs)
    if pretrained:
        if model_weights is None:
            raise AssertionError("No default weights to load")
        else:
            model.load_state_dict(model_weights)

    return model
