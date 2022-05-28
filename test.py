import hydra
from omegaconf import DictConfig
from utils.augmentation import get_aug
from utils.data_utils import TestDataset, DataLoader
from utils.batch_samplers import get_sampler
from utils.models import get_model
from utils.test_time_mapping import mapping
from tqdm import tqdm
import torch
import os


class Evaluator:
    def __init__(self, config: DictConfig):
        self.config = config['version']
        self.data_dir = config['Data']['data directory']
        self.datasets = config['Data']['test datasets']
        self.transform = get_aug(config['version']['Transform'], 'test')
        self.save_dir = config['version']['Experiment']['logs directory']
        self._init_dataloader()
        self._init_model()
        self.evaluate()

    def _init_model(self):
        self.model = get_model(self.config['Model']['architecture'], classes=self.config['mapping'])
        params = os.path.join(self.save_dir, 'best_model.params')
        checkpoint = torch.load(params)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.cuda()
        self.model.eval()

    def _init_dataloader(self) -> None:
        self.test_dataloaders = {}
        for dataset in self.datasets:
            # Get test dataset
            test_dataset = TestDataset(self.data_dir + '/' + dataset['name'],
                                       self.config['mapping'], transform=self.transform)
            # Get sampler: oversampling or None
            test_sampler = get_sampler(self.config, test_dataset, 'test')
            # Get test iterator
            loader = DataLoader(test_dataset, self.config['Parameters']['batch size'],
                                sampler=test_sampler)
            self.test_dataloaders[dataset['name']] = {
                'loader': loader,
                'iters': len(loader),
            }

    def evaluate(self) -> None:
        for test_name in self.test_dataloaders:
            loader = self.test_dataloaders[test_name]['loader']
            iters = self.test_dataloaders[test_name]['iters']
            iterator = tqdm(loader, total=iters, unit='batch', desc=test_name)
            for images, labels in iterator:
                labels = labels.cuda(non_blocking=True)
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    predictions = self.model(images)
                    # Test-time mapping
                    predictions = mapping(
                        predictions,
                        self.config['Model']['Mapping']['test_time_mapping'],
                        [attribute['name'] for attribute in self.config['mapping']]
                    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start_test(cfg: DictConfig) -> None:
    tester = Evaluator(cfg)


if __name__ == "__main__":
    start_test()
