from torch.nn.init import kaiming_normal_, zeros_
from utils.models.mobilnet import get_mobilenet
from utils.models.layers import OutputUnit
from omegaconf import DictConfig
import torch.nn as nn
import sys


def _weights_init_classifier(m):
    """ Initialize only Linear layers"""
    if isinstance(m, nn.Linear):
        kaiming_normal_(m.weight.data, mode='fan_out')
        if m.bias is not None:
            zeros_(m.bias.data)


def get_classifier(params: DictConfig, classes: list) -> nn.Module:
    output_type = params['output_type']
    in_channels = params['in_channels']
    classifier = nn.ModuleList([OutputUnit(output_type, in_channels=in_channels,
                                embedding_dim=class_info['length'])
                                for class_info in classes])
    return classifier


class Model(nn.Module):
    def __init__(self, backbone, classifier):
        super(Model, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        # input
        x = self.backbone(x)
        # output
        x = [cls(x) for cls in self.classifier]
        return x


def get_model(params: DictConfig, classes: list) -> nn.Module:
    # Create extractor model
    model = None
    if params['name'] == 'Mbnet':
        backbone = get_mobilenet(weights=params['init'])
        classifier = get_classifier(params, classes)
        model = Model(backbone, classifier)
    else:
        sys.exit('Unsupported net architecture: %s !' % params['type'])

    # Init Classifier
    model.apply(_weights_init_classifier)
    return model
