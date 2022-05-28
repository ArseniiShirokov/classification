from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import json
import torch
from torch import nn
import os


def save_module_state(module: nn.Module, filename: str) -> None:
    save_state = {
        'state_dict': module.state_dict()
    }
    torch.save(save_state, filename)


def save_best_model(checkpoint_model, save_dir: str, epoch_num: int) -> None:
    params = f'{save_dir}/model-{epoch_num:03}.params'
    dst_model_path = f'{save_dir}/best_model.pth'
    dst_params_path = f'{save_dir}/best_model.params'
    checkpoint = torch.load(params)
    checkpoint_model.load_state_dict(checkpoint["state_dict"])
    torch.save(checkpoint_model, dst_model_path)
    save_module_state(checkpoint_model, dst_params_path)


def save_config(config: DictConfig) -> None:
    save_filename = config['version']['Experiment']['logs directory'] + '/config.json'
    with open(save_filename, 'w') as f:
        json.dump(OmegaConf.to_container(config), f, sort_keys=True, indent=4)


def log_losses(losses: list, attributes: list) -> str:
    out_str = ''
    for i, loss in enumerate(losses):
        out_str += f'Loss{attributes[i]}: {loss.cur:.3f} ({loss.avg:.3f})\t'
    return out_str


class Logger:
    def __init__(self, experiment_dir: str):
        # duplicate logging to file and stdout
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s]\t%(message)s',
                            datefmt='%m-%d-%y %H:%M',
                            filename=experiment_dir + '/log.txt',
                            filemode='w')
        console = logging.StreamHandler()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console)

    def log(self, string: str) -> None:
        self.logger.info(string)


class WAndBLogger:
    def __init__(self, account_key: str, run_offline: bool, **init_params):
        # set environment
        os.environ["WANDB_API_KEY"] = account_key
        if run_offline:
            os.environ["WANDB_MODE"] = "offline"
        # init wandb run
        self.logger = wandb.init(**init_params)
        # init log state
        self.state = {}

    def update_state(self, data: dict) -> None:
        self.state.update(data)

    def log(self) -> None:
        self.logger.log(self.state)
        self.state = {}


def get_wandb_logger(config: DictConfig):
    if not config['enabled']:
        return None

    return WAndBLogger(config['account_key'],
                       config['run_offline'],
                       project=config['run']['project'],
                       name=config['run']['name'],
                       tags=config['run']['tags'],
                       group=config['run']['group'],
                       dir='logs')
