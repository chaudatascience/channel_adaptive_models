import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import MyConfig
from trainer import Trainer

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(version_base=None, config_path="configs/cifar", config_name="debug")
def main(cfg: MyConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # print(OmegaConf.to_container(cfg))

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
