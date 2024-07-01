from copy import deepcopy as dp
from pathlib import Path

from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(
        self,
        name,
        dataset_name,
        dataset_dir,
        trajectory,
        feature_type,
        num_rawfeats,
        num_feats,
        num_cams,
        num_cond_feats,
        standardization,
        augmentation=None,
        **modalities,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.name = name
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams
        self.trajectory_dataset = trajectory
        self.standardization = standardization
        self.modality_datasets = modalities

        if augmentation is not None:
            self.augmentation = True
            self.augmentation_rate = augmentation.rate
            self.trajectory_dataset.set_augmentation(augmentation.trajectory)
            if hasattr(augmentation, "modalities"):
                for modality, augments in augmentation.modalities:
                    self.modality_datasets[modality].set_augmentation(augments)
        else:
            self.augmentation = False

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split

        # Get trajectory split
        self.trajectory_dataset = dp(self.trajectory_dataset).set_split(
            split, train_rate
        )
        self.root_filenames = self.trajectory_dataset.filenames

        # Get modality split
        for modality_name in self.modality_datasets.keys():
            self.modality_datasets[modality_name].filenames = self.root_filenames

        self.get_feature = self.trajectory_dataset.get_feature
        self.get_matrix = self.trajectory_dataset.get_matrix

        return self

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index):
        traj_out = self.trajectory_dataset[index]
        traj_filename, traj_feature, padding_mask, intrinsics = traj_out

        out = {
            "traj_filename": traj_filename,
            "traj_feat": traj_feature,
            "padding_mask": padding_mask,
            "intrinsics": intrinsics,
        }

        for modality_name, modality_dataset in self.modality_datasets.items():
            modality_filename, modality_feature, modality_raw = modality_dataset[index]
            assert traj_filename.split(".")[0] == modality_filename.split(".")[0]
            out[f"{modality_name}_filename"] = modality_filename
            out[f"{modality_name}_feat"] = modality_feature
            out[f"{modality_name}_raw"] = modality_raw
            out[f"{modality_name}_padding_mask"] = padding_mask

        return out

    def __len__(self):
        return len(self.trajectory_dataset)
