from copy import deepcopy
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.datasets.datamodule import Datamodule
from src.callbacks.fix_nans import FixNANinGrad
from utils.random_utils import set_random_seed
import torch

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    set_random_seed(config.seed)

    dict_config = OmegaConf.to_container(config, resolve=True)

    if config.log_wandb:
        logger = WandbLogger(
            entity=config.entity,
            project=config.project_name,
            name=config.xp_name,
            save_dir=config.log_dir,
        )
        logger._wandb_init.update({"config": dict_config})
    else:
        logger = None

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config.save_and_sample_every, save_top_k=-1
    )
    fix_nan = FixNANinGrad(monitor=["train/loss"])

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, fix_nan, lr_monitor]

    trainer = instantiate(config.trainer)(logger=logger, callbacks=callbacks)

    diffuser = instantiate(config.diffuser)

    dataset = instantiate(config.dataset)
    datamodule = Datamodule(
        deepcopy(dataset).set_split("train"),
        deepcopy(dataset).set_split("test"),
        config.batch_size,
        config.compnode.num_workers,
    )

    trainer.fit(model=diffuser, datamodule=datamodule, ckpt_path=config.checkpoint_path)


if __name__ == "__main__":
    main()
