from utils.batch_samplers.onlabel_sampler import ImbalancedDatasetSampler
from utils.batch_samplers.multilabel_sampler import CastomDatasetSampler
from omegaconf import DictConfig


def get_sampler(params: DictConfig, dataset, _type='train'):
    oversampling = params['Model']['oversampling'][_type]
    attributes_cnt = len(params['mapping'])
    batch_size = params['Parameters']['batch size']

    sampler = None
    if oversampling:
        if attributes_cnt == 1:
            sampler = ImbalancedDatasetSampler(dataset)
        else:
            sampler = CastomDatasetSampler(dataset, attributes_cnt, batch_size, balanced=True)
    return sampler
