from pathlib import Path
import random

from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
import torch.nn.functional as F
from typing import Tuple

from utils.file_utils import load_txt
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d


# ------------------------------------------------------------------------------------- #

num_cams = None

# ------------------------------------------------------------------------------------- #


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        name: str,
        set_name: str,
        dataset_dir: str,
        num_rawfeats: int,
        num_feats: int,
        num_cams: int,
        standardize: bool,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.set_name = set_name
        self.dataset_dir = Path(dataset_dir)
        self.data_dir = self.dataset_dir / "traj"
        self.intrinsics_dir = self.dataset_dir / "intrinsics"

        self.num_rawfeats = num_rawfeats
        self.num_feats = num_feats
        self.num_cams = num_cams

        self.augmentation = None
        self.standardize = standardize
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean"])
            self.norm_std = torch.Tensor(mean_std["norm_std"])
            self.shift_mean = torch.Tensor(mean_std["shift_mean"])
            self.shift_std = torch.Tensor(mean_std["shift_std"])
            self.velocity = mean_std["velocity"]

    # --------------------------------------------------------------------------------- #

    def set_split(self, split: str, train_rate: float = 1.0):
        self.split = split
        split_path = Path(self.dataset_dir) / f"{self.set_name}_{split}_split.txt"
        split_traj = load_txt(split_path).split("\n")

        if self.split == "train":
            train_size = int(len(split_traj) * train_rate)
            train_split_traj = random.sample(split_traj, train_size)
            self.filenames = sorted(train_split_traj)
        else:
            self.filenames = sorted(split_traj)

        return self

    # --------------------------------------------------------------------------------- #

    def get_feature(
        self, raw_matrix_trajectory: TensorType["num_cams", 4, 4]
    ) -> TensorType[9, "num_cams"]:
        matrix_trajectory = torch.clone(raw_matrix_trajectory)

        raw_trans = torch.clone(matrix_trajectory[:, :3, 3])
        if self.velocity:
            velocity = raw_trans[1:] - raw_trans[:-1]
            raw_trans = torch.cat([raw_trans[0][None], velocity])
        if self.standardize:
            raw_trans[0] -= self.shift_mean
            raw_trans[0] /= self.shift_std
            raw_trans[1:] -= self.norm_mean
            raw_trans[1:] /= self.norm_std

        # Compute the 6D continuous rotation
        raw_rot = matrix_trajectory[:, :3, :3]
        rot6d = raw_rot[:, :, :2].permute(0, 2, 1).reshape(-1, 6)

        # Stack rotation 6D and translation
        rot6d_trajectory = torch.hstack([rot6d, raw_trans]).permute(1, 0)

        return rot6d_trajectory

    def get_matrix(
        self, raw_rot6d_trajectory: TensorType[9, "num_cams"]
    ) -> TensorType["num_cams", 4, 4]:
        rot6d_trajectory = torch.clone(raw_rot6d_trajectory)
        device = rot6d_trajectory.device

        num_cams = rot6d_trajectory.shape[1]
        matrix_trajectory = torch.eye(4, device=device)[None].repeat(num_cams, 1, 1)

        raw_trans = rot6d_trajectory[6:].permute(1, 0)
        if self.standardize:
            raw_trans[0] *= self.shift_std.to(device)
            raw_trans[0] += self.shift_mean.to(device)
            raw_trans[1:] *= self.norm_std.to(device)
            raw_trans[1:] += self.norm_mean.to(device)
        if self.velocity:
            raw_trans = torch.cumsum(raw_trans, dim=0)
        matrix_trajectory[:, :3, 3] = raw_trans

        rot6d = rot6d_trajectory[:6].permute(1, 0)
        raw_rot = compute_rotation_matrix_from_ortho6d(rot6d)
        matrix_trajectory[:, :3, :3] = raw_rot

        return matrix_trajectory

    # --------------------------------------------------------------------------------- #

    def __getitem__(self, index: int) -> Tuple[str, TensorType["num_cams", 4, 4]]:
        filename = self.filenames[index]

        trajectory_filename = filename + ".txt"
        trajectory_path = self.data_dir / trajectory_filename

        trajectory = read_kitti_poses_file(trajectory_path)
        matrix_trajectory = torch.from_numpy(np.array(trajectory.poses_se3)).to(
            torch.float32
        )

        intrinsics_filename = filename + ".npy"
        intrinsics_path = self.intrinsics_dir / intrinsics_filename
        intrinsics = np.load(intrinsics_path)

        trajectory_feature = self.get_feature(matrix_trajectory)

        padded_trajectory_feature = F.pad(
            trajectory_feature, (0, self.num_cams - trajectory_feature.shape[1])
        )
        # Padding mask: 1 for valid cams, 0 for padded cams
        padding_mask = torch.ones((self.num_cams))
        padding_mask[trajectory_feature.shape[1] :] = 0

        return (
            trajectory_filename,
            padded_trajectory_feature,
            padding_mask,
            intrinsics,
        )

    def __len__(self):
        return len(self.filenames)
