from torch.utils.data import Dataset, DataLoader
from utils.batch_samplers import get_sampler
from utils.augmentation import get_aug
from omegaconf import DictConfig
from typing import Tuple
import pandas as pd
import mxnet as mx
import numpy as np
import torch


def get_data_iterators(config: DictConfig, data: DictConfig) -> Tuple:
    val_enabled = 'val dataset' in config['Data']
    # Set aug transformations
    transform_train = get_aug(config['Transform'], 'train')
    simple_transform = get_aug(config['Transform'], 'test')
    # Create train dataset
    dataset_train = TrainDataset(data['data directory'] + '/' + data['train dataset'],
                                 config['mapping'], config['Transform']['jsd']['enabled'],
                                 simple_transform=simple_transform, transform=transform_train)
    # Get sampler: oversampling or None
    train_sampler = get_sampler(config, dataset_train, 'train')
    # Shuffle doesn't need with sampler
    train_shuffle = True if train_sampler is None else False
    # Create train iterator
    train_iter = DataLoader(dataset_train, sampler=train_sampler,
                            batch_size=config['Parameters']['batch size'],
                            num_workers=config['Parameters']['num workers'], shuffle=train_shuffle)

    val_iter = None
    if val_enabled:
        # Set aug transformations
        transform_val = get_aug(config['Transform'], 'val')
        # Create val dataset
        dataset_val = TestDataset(data['data directory'] + '/' + data['val dataset'],
                                  config['mapping'], transform=transform_val)
        # Get sampler: oversampling or None
        val_sampler = get_sampler(config, dataset_val, 'val')
        # Shuffle doesn't need with sampler
        val_shuffle = True if val_sampler is None else False
        # Create validation iterator
        val_iter = DataLoader(dataset_val, sampler=val_sampler,
                              batch_size=config['Parameters']['batch size'], num_workers=1, shuffle=val_shuffle)

    weights = get_weights(config, data)
    return train_iter, val_iter, weights


def get_weights(config: DictConfig, data: DictConfig,no_clip=False):
    train_data = pd.read_csv(data['data directory'] + '/' + data['train dataset'] + '.csv')
    attributes = config['mapping']

    for i, attribute in enumerate(attributes):
        attributes[i]['length'] = len(attribute['values'])

    weights = []
    for attribute in attributes:
        values = [value for value in train_data[attribute['name']] if value not in [-1]]
        values = np.array(values, dtype=np.int32)
        num_values = attribute['length']
        if no_clip or not config['Model']['loss']['weights']:
            # Compute class probability for thresholding
            counts = np.bincount(values, minlength=num_values)
            _weights = values.size / (num_values * counts + 1e-6)
        else:
            counts = np.bincount(values, minlength=num_values)
            _weights = values.size / (num_values * counts + 1e-6)
            _weights = np.clip(_weights, config['Model']['loss']['weight params']['min weight'],
                               config['Model']['loss']['weight params']['max weight'])
        _weights = torch.tensor(_weights, dtype=torch.float)
        weights.append(_weights)
    return weights


class TrainDataset(Dataset):
    def __init__(self, root_dir, mapping: DictConfig, use_jsd=False, transform=None, simple_transform=None):
        self.attributes = [attribute['name'] for attribute in mapping]
        self.root_dir = root_dir
        self.imgrec = mx.recordio.MXIndexedRecordIO(self.root_dir + '.idx',
                                                    self.root_dir + '.rec', 'r')
        self.transform = transform
        self.use_jsd = use_jsd
        self.mask = self.init_filter_mask(mapping)
        if simple_transform:
            self.simple_transform = simple_transform
        else:
            self.simple_transform = transform

    def init_filter_mask(self, mapping: DictConfig):
        all_attributes = pd.read_csv(self.root_dir + '.csv').columns[1:]
        return [attribute in self.attributes for attribute in all_attributes]

    def __len__(self):
        return pd.read_csv(self.root_dir + '.csv').shape[0]

    def __getitem__(self, idx):
        data = self.imgrec.read_idx(idx)
        header, image = mx.image.recordio.unpack(data)
        labels = torch.tensor(header.label, dtype=torch.long)[self.mask]
        img = mx.image.imdecode(image).asnumpy()

        image = self.simple_transform(image=img)
        image_aug1 = self.transform(image=img)
        image_aug2 = self.transform(image=img)

        image = image['image']
        image_aug1 = image_aug1['image']
        image_aug2 = image_aug2['image']

        image = torch.tensor(image, dtype=torch.float).permute((2, 0, 1))
        image_aug1 = torch.tensor(image_aug1, dtype=torch.float).permute((2, 0, 1))
        image_aug2 = torch.tensor(image_aug2, dtype=torch.float).permute((2, 0, 1))
        if self.use_jsd:
            return torch.stack([image, image_aug1, image_aug2]), labels
        else:
            return image_aug1, labels

    def get_labels(self):
        """Needed for oversampling"""
        data = pd.read_csv(self.root_dir + '.csv')
        if len(self.attributes) == 1:
            return data[self.attributes[0]]
        else:
            data = data[self.attributes]
            return data


class TestDataset(Dataset):
    def __init__(self, root_dir, mapping: DictConfig, transform=None):
        self.root_dir = root_dir
        self.imgrec = mx.recordio.MXIndexedRecordIO(self.root_dir + '.idx',
                                                    self.root_dir + '.rec', 'r')
        self.transform = transform
        self.mask = self.init_filter_mask(mapping)

    def init_filter_mask(self, mapping: DictConfig):
        all_attributes = pd.read_csv(self.root_dir + '.csv').columns[1:]
        needed_attributes = [attribute['name'] for attribute in mapping]
        return [attribute in needed_attributes for attribute in all_attributes]

    def __len__(self):
        return pd.read_csv(self.root_dir + '.csv').shape[0]

    def __getitem__(self, idx):
        data = self.imgrec.read_idx(idx)
        header, image = mx.image.recordio.unpack(data)
        labels = torch.tensor(header.label, dtype=torch.long)[self.mask]
        img = mx.image.imdecode(image).asnumpy()
        image = self.transform(image=img)
        image = image['image']
        image = torch.tensor(image, dtype=torch.float).permute((2, 0, 1))
        return image, labels
