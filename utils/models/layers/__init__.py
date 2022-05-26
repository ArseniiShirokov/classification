import torch.nn as nn
import sys


class Activation(nn.Module):
    def __init__(self, act_type, channels=1):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.activation = nn.PReLU(channels)
        else:
            sys.exit('Wrong activation type: %s !' % act_type)

    def forward(self, x):
        x = self.activation(x)
        return x


class OutputUnit(nn.Module):
    def __init__(self, output_type, in_channels, embedding_dim=512):
        super(OutputUnit, self).__init__()
        layers = []
        if output_type == 'Demography':
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(in_channels, embedding_dim))
        else:
            sys.exit('Unsupported output unit type: %s' % output_type)
        self.output_unit = nn.Sequential(*layers)

    def forward(self, x):
        x = self.output_unit(x)
        return x
