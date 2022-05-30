from omegaconf import DictConfig
import hydra
from train import start_train
from test import start_test


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start_train_test(cfg: DictConfig) -> None:
    start_train(cfg)
    start_test(cfg)


if __name__ == "__main__":
    start_train_test()
