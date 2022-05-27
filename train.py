from torch.nn.parallel import DistributedDataParallel as DDP
from utils.data_utils import get_data_iterators
from utils.train_utils import AverageMeter
from utils.optimizer import get_optimizer
from utils.optimizer import get_scheduler
from utils.logger import log_losses
from utils.logger import get_wandb_logger
from utils.logger import save_config
from utils.models import get_model
from utils.losses import get_loss
from omegaconf import DictConfig
import torch.distributed as dist
from utils.logger import Logger
from utils.logger import save_module_state
from utils.logger import save_best_model
from typing import Tuple
from utils.forwards import simple_forward
from utils.forwards import jsd_forward
import random
import shutil
import numpy
import hydra
import torch
import time
import sys
import os


class Trainer:
    def __init__(self, rank: int, config: DictConfig, wandb: DictConfig, world_size: int) -> None:
        self.use_jsd = config['Transform']['jsd']['enabled']
        self.attributes = [attribute['name'] for attribute in config['mapping']]
        self.config = config
        self._init_ddp(rank, config, world_size)
        # if we want full reproducible, deterministic results
        if config['Parameters']['deterministic']:
            self._init_random_seed(42 + rank)
        self._init_loggers(config, wandb)
        self._init_training_params(config)
        self.train()

    # ============== Initialization helpers ==============
    def _init_ddp(self, rank: int, config: DictConfig, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        dist.init_process_group(backend=config['Parameters']['ddp backend'],
                                rank=self.rank,
                                world_size=self.world_size)

    def _init_training_params(self, config: DictConfig) -> None:
        self.attributes_cnt = len(config['mapping'])
        self._init_data_iterators(config)
        self._init_criterion(config)
        self._init_model(config)
        self._init_optimizer(config)

        self.best_epoch = 0
        self.save_dir = config['Experiment']["logs directory"]
        self.display_period = config['Experiment']['display period']
        self.device_ids = config['Parameters']['context device ids']
        self.device_batchsize = config['Parameters']['batch size']
        self.total_batchsize = self.device_batchsize * self.world_size
        self.start_epoch = 0
        self.end_epoch = config['Parameters']['num epochs']

    def _init_data_iterators(self, config: DictConfig) -> None:
        self.train_iter, self.val_iter, self.weights = get_data_iterators(config)
        self.train_iter_len = len(self.train_iter)
        if self.val_iter:
            self.val_iter_len = len(self.val_iter)

    def _init_loggers(self, config: DictConfig, wandb: DictConfig) -> None:
        self.save_dir = config['Experiment']['logs directory']
        save_config(config)
        if self.rank == 0:
            self.logger = Logger(self.save_dir)
            self.wandb = get_wandb_logger(wandb)

    def _init_criterion(self, config: DictConfig) -> None:
        weights = []
        for i, weight in enumerate(self.weights):
            new = weight.to(self.rank) if config['Model']['loss']['weights'] else None
            weights.append(new)
        loss = config['Model']['loss']
        self.criterion = [get_loss(loss, weights[i]) for i in range(self.attributes_cnt)]

    def _init_model(self, config: DictConfig) -> None:
        model = get_model(config['Model']['architecture'], classes=config['mapping'])
        self.checkpoint_model = get_model(config['Model']['architecture'], classes=config['mapping'])
        # Freeze backbone
        for module in [model.backbone]:
            for name, param in module.named_parameters():
                param.requires_grad = not config['Model']['architecture']['freeze']
        # DDP
        model = model.to(self.rank)
        self.model = DDP(model, device_ids=[self.rank])

    def _init_optimizer(self, config: DictConfig) -> None:
        model_named_params = self.model.named_parameters()
        # Mixed precision
        self.amp = config['Parameters']['amp']
        # Create optimizer, lr-scheduler, and amp-grad-scaler
        self.optimizer = get_optimizer(model_named_params,
                                       config['Parameters'])
        self.scheduler = get_scheduler(self.optimizer,
                                       config['Parameters']['scheduler'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    @staticmethod
    def _init_random_seed(seed: int) -> None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =============== Logging helpers ===============
    def log(self, string: str) -> None:
        if self.rank == 0:
            self.logger.log(string)

    def _wandb_update_state(self, info: dict) -> None:
        if self.rank == 0 and self.wandb:
            self.wandb.update_state(info)

    def _wandb_log(self) -> None:
        if self.rank == 0 and self.wandb:
            self.wandb.log()

    def _save_state(self, epoch: int) -> None:
        if self.rank == 0:
            # Save extractor snapshot
            filename = '{}/model-{:03d}.params'.format(
                self.save_dir, epoch)
            save_module_state(self.model.module, filename)

    # ================= Training =================
    def train(self) -> None:
        # Switch to train mode
        self.model.train()
        # Loop over epochs
        for epoch in range(self.start_epoch, self.end_epoch):
            tic_epoch = time.time()
            self.train_epoch(epoch)
            epoch_time = time.time() - tic_epoch
            self.log('Epoch: [{}] Total time: {:.3f} seconds'.format(
                epoch, epoch_time))
            self.scheduler.step()
        # Save best model
        save_best_model(self.checkpoint_model, self.save_dir, self.best_epoch)
        # Terminate DDP
        dist.destroy_process_group()
        # Stop if not stoped
        sys.exit(0)

    def train_epoch(self, epoch: int) -> None:
        # Init counters
        batch_time = AverageMeter()
        losses = [AverageMeter() for _ in range(len(self.config['mapping']))]
        # Training loop
        tic_batch = time.time()
        for i, batch in enumerate(self.train_iter):
            # Do training iteration
            data, labels = batch
            data.to(self.rank, non_blocking=True)
            labels.squeeze().long().to(self.rank, non_blocking=True)
            logits, loss = self._train_step(data, labels)
            for idx, _loss in enumerate(loss):
                losses[idx].update(_loss.item(), self.total_batchsize)
            self._wandb_update_state({'epoch': epoch})
            for k, attribute in enumerate(self.attributes):
                self._wandb_update_state({f'{attribute}-train-loss': loss[k].item()})
            if self.display_period and not (i + 1) % self.display_period:
                learning_rate = self.scheduler.get_last_lr()[0]
                torch.cuda.synchronize()
                batch_time.update((time.time() - tic_batch) /
                                  self.display_period)
                tic_batch = time.time()
                self.log(f'Iter: [{epoch}/{self.end_epoch}][{i + 1}/{self.train_iter_len}]\n'
                         f'Time: {batch_time.cur:.3f} ({batch_time.avg:.3f})\n'
                         f'LR: {learning_rate:.6f}\n'
                         f'{log_losses(losses, self.attributes)}\n\n')
            self._wandb_log()
        self._save_state(epoch)

    def _train_step(self, data: torch.Tensor, labels: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # forward
        logits, loss = self._forward(data, labels)
        # backward
        self.scaler.scale(self._sum_losses(loss)).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        # update grad scaler
        self.scaler.update()
        return logits, loss

    def _forward(self, data: torch.Tensor, labels: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.amp):
            if self.use_jsd:
                with_jsd_att = self.config['Transform']['jsd']['attributes']
                mask = [i for i, attribute in enumerate(self.attributes) if attribute in with_jsd_att]
                return jsd_forward(self.model, data, labels, self.criterion, mask)
            else:
                return simple_forward(self.model, data, labels, self.criterion)

    @staticmethod
    def _sum_losses(loss: list) -> torch.Tensor:
        loss_sum = torch.tensor(0, device=loss[0].device, dtype=torch.float32)
        for i, _loss in enumerate(loss):
            loss_sum += _loss
        return loss_sum


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start_train(cfg: DictConfig) -> None:
    wandb = cfg['Wandb']
    cfg = cfg['version']

    # Create dir to save exps
    if os.path.exists(cfg['Experiment']['logs directory']):
        val = input("Warning! Dir is exists, continue? y/n \n")
        if val != 'y':
            return
        shutil.rmtree(cfg['Experiment']['logs directory'])
    os.makedirs(cfg['Experiment']['logs directory'])

    # Do some preparation stuff for DistributedDataParallel
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    # Start training
    trainer = Trainer
    world_size = len(cfg['Parameters']['context device ids'])
    torch.multiprocessing.spawn(trainer,
                                args=(cfg, wandb, world_size,),
                                nprocs=world_size,
                                join=True
                                )


if __name__ == "__main__":
    start_train()
