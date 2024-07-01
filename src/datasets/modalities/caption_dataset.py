from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.file_utils import load_txt


class CaptionDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_cams: int,
        num_feats: int,
        num_segments: int,
        sequential: bool,
        **kwargs,
    ):
        super().__init__()
        self.modality = name
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        # Set data paths (segments, captions, etc...)
        for name, field in kwargs.items():
            if isinstance(field, str):
                field = Path(field)
            if name == "feat_caption_dir":
                field = field / "seq" if sequential else field / "token"
            setattr(self, name, field)

        self.filenames = None

        self.clip_seq_dir = self.dataset_dir / "caption_clip" / "seq"  # For CLaTrScore
        self.num_cams = num_cams
        self.num_feats = num_feats
        self.num_segments = num_segments
        self.sequential = sequential

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Load data
        if hasattr(self, "segment_dir"):
            raw_segments = torch.from_numpy(
                np.load((self.segment_dir / (filename + ".npy")))
            )
            padded_raw_segments = F.pad(
                raw_segments,
                (0, self.num_cams - len(raw_segments)),
                value=self.num_segments,
            )
        if hasattr(self, "raw_caption_dir"):
            raw_caption = load_txt(self.raw_caption_dir / (filename + ".txt"))
        if hasattr(self, "feat_caption_dir"):
            feat_caption = torch.from_numpy(
                np.load((self.feat_caption_dir / (filename + ".npy")))
            )
            if self.sequential:
                feat_caption = F.pad(
                    feat_caption.to(torch.float32),
                    (0, 0, 0, self.max_feat_length - feat_caption.shape[0]),
                )

        if self.modality == "caption":
            raw_data = {"caption": raw_caption, "segments": padded_raw_segments}
            feat_data = (
                feat_caption.permute(1, 0) if feat_caption.dim() == 2 else feat_caption
            )
        elif self.modality == "segments":
            raw_data = {"segments": padded_raw_segments}
            # Shift by one for padding
            feat_data = F.one_hot(
                padded_raw_segments, num_classes=self.num_segments + 1
            ).to(torch.float32)
            if self.sequential:
                feat_data = feat_data.permute(1, 0)
            else:
                feat_data = feat_data.reshape(-1)
        elif self.modality == "class":
            raw_data = {"segments": padded_raw_segments}
            most_frequent_segment = Counter(raw_segments).most_common(1)[0][0]
            feat_data = F.one_hot(
                torch.tensor(most_frequent_segment), num_classes=self.num_segments
            ).to(torch.float32)
        else:
            raise ValueError(f"Modality {self.modality} not supported")

        clip_seq_caption = torch.from_numpy(
            np.load((self.clip_seq_dir / (filename + ".npy")))
        )
        padding_mask = torch.ones((self.max_feat_length))
        padding_mask[clip_seq_caption.shape[0] :] = 0
        clip_seq_caption = F.pad(
            clip_seq_caption.to(torch.float32),
            (0, 0, 0, self.max_feat_length - clip_seq_caption.shape[0]),
        )
        raw_data["clip_seq_caption"] = clip_seq_caption
        raw_data["clip_seq_mask"] = padding_mask

        return filename, feat_data, raw_data
