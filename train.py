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
    def __init__(self, rank: int, config: DictConfig, world_size: int) -> None:
        self._init_loggers(config, config['Wandb'])
        self.config = config['version']
        self.data = config["Data"]
        self.use_jsd = self.config['Transform']['jsd']['enabled']
        self.attributes = [attribute['name'] for attribute in self.config['mapping']]
        self._init_ddp(rank, world_size)
        # if we want full reproducible, deterministic results
        if self.config['Parameters']['deterministic']:
            self._init_random_seed(42 + rank)
        self._init_training_params()
        self.train()

    # ============== Initialization helpers ==============
    def _init_ddp(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        dist.init_process_group(backend=self.config['Parameters']['ddp backend'],
                                rank=self.rank,
                                world_size=self.world_size)

    def _init_training_params(self) -> None:
        self.attributes_cnt = len(self.config['mapping'])
        self._init_data_iterators()
        self._init_criterion()
        self._init_model()
        self._init_optimizer()
        self.best_epoch = 0
        self.current_best_loss = None
        self.save_dir = self.config['Experiment']["logs directory"]
        self.display_period = self.config['Experiment']['display period']
        self.device_ids = self.config['Parameters']['context device ids']
        self.device_batchsize = self.config['Parameters']['batch size']
        self.total_batchsize = self.device_batchsize * self.world_size
        self.start_epoch = 0
        self.end_epoch = self.config['Parameters']['num epochs']

    def _init_data_iterators(self) -> None:
        self.train_iter, self.val_iter, self.weights = get_data_iterators(self.config, self.data)
        self.train_iter_len = len(self.train_iter)
        if self.val_iter:
            self.val_iter_len = len(self.val_iter)

    def _init_loggers(self, config: DictConfig, wandb: DictConfig) -> None:
        self.save_dir = config['version']['Experiment']['logs directory']
        save_config(config)
        if self.rank == 0:
            self.logger = Logger(self.save_dir)
            self.wandb = get_wandb_logger(wandb)

    def _init_criterion(self) -> None:
        weights = []
        for i, weight in enumerate(self.weights):
            new = weight.to(self.rank) if self.config['Model']['loss']['weights'] else None
            weights.append(new)
        loss = self.config['Model']['loss']
        self.criterion = [get_loss(loss, weights[i]) for i in range(self.attributes_cnt)]

    def _init_model(self) -> None:
        model = get_model(self.config['Model']['architecture'], classes=self.config['mapping'])
        self.checkpoint_model = get_model(self.config['Model']['architecture'], classes=self.config['mapping'])
        # Freeze backbone
        for module in [model.backbone]:
            for name, param in module.named_parameters():
                param.requires_grad = not self.config['Model']['architecture']['freeze']
        # DDP
        model = model.to(self.rank)
        self.model = DDP(model, device_ids=[self.rank])

    def _init_optimizer(self) -> None:
        model_named_params = self.model.named_parameters()
        # Mixed precision
        self.amp = self.config['Parameters']['amp']
        # Create optimizer, lr-scheduler, and amp-grad-scaler
        self.optimizer = get_optimizer(model_named_params,
                                       self.config['Parameters'])
        self.scheduler = get_scheduler(self.optimizer,
                                       self.config['Parameters']['scheduler'])
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
            self._train_epoch(epoch)
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

    def _train_epoch(self, epoch: int) -> None:
        # Init counters
        batch_time = AverageMeter()
        losses = [AverageMeter() for _ in range(len(self.attributes))]
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
                batch_time.update((time.time() - tic_batch) /
                                  self.display_period)
                tic_batch = time.time()
                self.log(f'Iter: [{epoch}/{self.end_epoch}][{i + 1}/{self.train_iter_len}]\n'
                         f'Time: {batch_time.cur:.3f} ({batch_time.avg:.3f})\n'
                         f'LR: {learning_rate:.6f}\n'
                         f'{log_losses(losses, self.attributes)}\n\n')
            self._wandb_log()
        self.validate(epoch)
        if epoch == self.best_epoch:
            self._save_state(epoch)

    def _train_step(self, data: torch.Tensor, labels: torch.Tensor) \
            -> Tuple[list, list]:
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
            -> Tuple[list, list]:
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

    # ================= Validation =================
    def validate(self, epoch: int):
        self.model.eval()
        self.log("Validation...")
        # Init counters
        losses = [AverageMeter() for _ in range(len(self.attributes))]
        # Val loop
        for i, batch in enumerate(self.val_iter):
            # Do val iteration
            data, labels = batch
            data.to(self.rank, non_blocking=True)
            labels.squeeze().long().to(self.rank, non_blocking=True)
            with torch.no_grad():
                logits, loss = self._forward(data, labels)
            for idx, _loss in enumerate(loss):
                losses[idx].update(_loss.item(), self.total_batchsize)
        # Compute criterion for find best model
        avg_loss = AverageMeter()
        for k, loss in enumerate(losses):
            avg_loss.update(loss.avg)
            self._wandb_update_state({f'{self.attributes[k]}-val-loss': loss.avg})
        # Update best epoch
        if self.current_best_loss is None or avg_loss.avg < self.current_best_loss:
            self.current_best_loss = avg_loss.avg
            self.best_epoch = epoch
        # log validation losses
        self.log('#' * 18)
        for i, loss in enumerate(losses):
            self.log(f'ValLoss - {self.attributes[i]}: ({loss.avg:.3f})\n')
        self.log('#' * 18)
        self.model.train()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start_train(cfg: DictConfig) -> None:
    # Create dir to save exps
    if os.path.exists(cfg['version']['Experiment']['logs directory']):
        val = input("Warning! Dir is exists, continue? y/n \n")
        if val != 'y':
            return
        shutil.rmtree(cfg['version']['Experiment']['logs directory'])
    os.makedirs(cfg['version']['Experiment']['logs directory'])

    # Do some preparation stuff for DistributedDataParallel
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    # Start training
    trainer = Trainer
    world_size = len(cfg['version']['Parameters']['context device ids'])
    torch.multiprocessing.spawn(trainer,
                                args=(cfg, world_size,),
                                nprocs=world_size,
                                join=True
                                )


if __name__ == "__main__":
    start_train()
