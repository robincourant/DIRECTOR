import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchtyping import TensorType

num_samples, num_feats = None, None


class CLaTrScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("traj_feat", default=[], dist_reduce_fx="cat")
        self.add_state("text_feats", default=[], dist_reduce_fx="cat")

    def update(
        self,
        traj_feat: TensorType["num_samples", "num_feats"],
        text_feats: TensorType["num_samples", "num_feats"],
    ) -> float:
        """Update state with new trajectory and text features."""
        self.traj_feat.append(traj_feat / traj_feat.norm(p=2, dim=-1, keepdim=True))
        self.text_feats.append(text_feats / text_feats.norm(p=2, dim=-1, keepdim=True))

    def compute(self) -> float:
        """Compute cosine similarity between trajectory and text features."""
        traj_feat = dim_zero_cat(self.traj_feat)
        text_feats = dim_zero_cat(self.text_feats)

        score = (100 * (traj_feat * text_feats).sum(axis=-1)).mean()

        return torch.max(score, torch.zeros_like(score))
