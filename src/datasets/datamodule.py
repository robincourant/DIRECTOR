from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class Datamodule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        batch_train_size: int,
        num_workers: int,
        eval_batch_size: int = None,
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.batch_train_size = batch_train_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_train_size
        )

        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        persistent_workers = True if self.num_workers > 0 else False

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_train_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        persistent_workers = True if self.num_workers > 0 else False

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        """Load predict set loader."""
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )
        return dataloader
