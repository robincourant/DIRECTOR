from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ------------------------------------------------------------------------------------- #


class CharacterDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        standardize: bool,
        num_feats: int,
        num_cams: int,
        sequential: bool,
        num_vertices: int,
        num_faces: int,
        load_vertices: bool,
        **kwargs,
    ):
        super().__init__()
        self.modality = "char"
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        self.traj_dir = self.dataset_dir / "traj"
        self.data_dir = self.dataset_dir / self.name
        self.vert_dir = self.dataset_dir / "vert_decimated"
        self.center_dir = self.dataset_dir / "char_raw"

        self.filenames = None
        self.standardize = standardize
        if self.standardize:
            mean_std = kwargs["standardization"]
            self.norm_mean = torch.Tensor(mean_std["norm_mean_h"])[:, None]
            self.norm_std = torch.Tensor(mean_std["norm_std_h"])[:, None]
            self.velocity = mean_std["velocity"]

        self.num_cams = num_cams
        self.num_feats = num_feats
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.sequential = sequential

        self.load_vertices = load_vertices

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        char_filename = filename + ".npy"
        char_path = self.data_dir / char_filename

        raw_char_feature = torch.from_numpy(np.load((char_path))).to(torch.float32)
        padding_size = self.num_cams - raw_char_feature.shape[0]
        padded_raw_char_feature = F.pad(
            raw_char_feature, (0, 0, 0, padding_size)
        ).permute(1, 0)

        center_path = self.center_dir / char_filename  # Center to offset mesh
        center_offset = torch.from_numpy(np.load(center_path)[0]).to(torch.float32)
        if self.load_vertices:
            vert_path = self.vert_dir / char_filename
            raw_verts = np.load(vert_path, allow_pickle=True)[()]
            if raw_verts["vertices"] is None:
                padded_verts = torch.zeros(
                    (self.num_cams, self.num_vertices, 3), dtype=torch.float32
                )
                padded_faces = torch.zeros(
                    (self.num_cams, self.num_faces, 3), dtype=torch.int16
                )
            else:
                verts = torch.from_numpy(raw_verts["vertices"]).to(torch.float32)
                faces = torch.from_numpy(raw_verts["faces"]).to(torch.int16)
                verts -= center_offset
                padded_verts = F.pad(verts, (0, 0, 0, 0, 0, padding_size))
                padded_faces = F.pad(faces, (0, 0, 0, 0, 0, padding_size))

        char_feature = raw_char_feature.clone()
        if self.velocity:
            velocity = char_feature[1:].clone() - char_feature[:-1].clone()
            char_feature = torch.cat([raw_char_feature[0][None], velocity])

        if self.standardize:
            # Normalize the first frame (orgin) and the rest (velocity) separately
            if len(self.norm_mean) == 6:
                char_feature[0] -= self.norm_mean[:3, 0].to(raw_char_feature.device)
                char_feature[0] /= self.norm_std[:3, 0].to(raw_char_feature.device)
                char_feature[1:] -= self.norm_mean[3:, 0].to(raw_char_feature.device)
                char_feature[1:] /= self.norm_std[3:, 0].to(raw_char_feature.device)
            # Normalize all in one
            else:
                char_feature -= self.norm_mean[:, 0].to(raw_char_feature.device)
                char_feature /= self.norm_std[:, 0].to(raw_char_feature.device)
        padded_char_feature = F.pad(
            char_feature,
            (0, 0, 0, self.num_cams - char_feature.shape[0]),
        )

        if self.sequential:
            padded_char_feature = padded_char_feature.permute(1, 0)
        else:
            padded_char_feature = padded_char_feature.reshape(-1)

        raw_feats = {"char_raw_feat": padded_raw_char_feature}
        raw_feats["char_centers"] = padded_raw_char_feature
        if self.load_vertices:
            raw_feats["char_vertices"] = padded_verts
            raw_feats["char_faces"] = padded_faces

        return char_filename, padded_char_feature, raw_feats
