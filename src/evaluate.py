from copy import deepcopy
from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.random_utils import set_random_seed

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    set_random_seed(config.seed)

    assert config.compnode.num_gpus == 1, "Evaluation script only supports single GPU"

    trainer = instantiate(config.trainer)()
    diffuser = instantiate(config.diffuser)
    dataset = instantiate(config.dataset)

    diffuser.ema.initted = torch.empty(()).to(diffuser.ema.initted)
    diffuser.ema.step = torch.empty(()).to(diffuser.ema.initted)
    diffuser.load_state_dict(
        torch.load(config.checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]
    )

    train_dataset = deepcopy(dataset).set_split("train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.compnode.num_workers,
        pin_memory=True,
    )
    test_dataset = deepcopy(dataset).set_split("test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.compnode.num_workers,
        pin_memory=True,
    )
    diffuser.get_matrix = train_dataset.get_matrix
    diffuser.modalities = list(train_dataset.modality_datasets.keys())

    metrics_dict = {}
    # Compute distribution metrics on the training set
    diffuser.metrics_to_compute = ["distribution"]
    trainer.test(model=diffuser, dataloaders=train_dataloader)
    metrics_dict["clatr/fcd"] = diffuser.metrics_dict["clatr/fcd"]
    metrics_dict["clatr/precision"] = diffuser.metrics_dict["clatr/precision"]
    metrics_dict["clatr/recall"] = diffuser.metrics_dict["clatr/recall"]
    metrics_dict["clatr/density"] = diffuser.metrics_dict["clatr/density"]
    metrics_dict["clatr/coverage"] = diffuser.metrics_dict["clatr/coverage"]

    # Compute semantic metrics on the test set
    diffuser.metrics_to_compute = ["distribution", "semantic"]
    trainer.test(model=diffuser, dataloaders=test_dataloader)
    metrics_dict["clatr/clatr_score"] = diffuser.metrics_dict["clatr/clatr_score"]
    metrics_dict["captions/precision"] = diffuser.metrics_dict["captions/precision"]
    metrics_dict["captions/recall"] = diffuser.metrics_dict["captions/recall"]
    metrics_dict["captions/fscore"] = diffuser.metrics_dict["captions/fscore"]

    metric_dir = Path(config.metric_dir)
    metric_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(config.checkpoint_path).stem
    metric_path = metric_dir / (config.xp_name + "-" + checkpoint_name + ".csv")
    metric_df = pd.DataFrame.from_dict(metrics_dict)
    metric_df.to_csv(metric_path, index=False)
    print(f"Metrics saved to {metric_path} \n")
    print(metric_df)


if __name__ == "__main__":
    main()
