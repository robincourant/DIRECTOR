"""
This code is adapted from https://github.com/Mathux/TMR
"""

from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
import lightning as L

from clatr.src.training.losses import KLLoss


def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class TEMOS(L.LightningModule):
    r"""
    Code highly adapated from:
    TEMOS: Generating diverse human motions
    from textual descriptions
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/temos

    :param traj_encoder: a module to encode the input traj features in the latent space.
    :param text_encoder: a module to encode the text embeddings in the latent space.
    :param traj_decoder: a module to decode the latent vector into traj features.
    :param vae: a boolean to make the model probabilistic.
    :param fact: a scaling factor for sampling the VAE (optional).
    :param sample_mean: sample the mean vector instead of random sampling (optional).
    :param lmd: dictionary of losses weights (optional).
    :param lr: learninig rate for the optimizer (optional).
    """

    def __init__(
        self,
        traj_encoder: nn.Module,
        text_encoder: nn.Module,
        traj_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5},
        lr: float = 1e-4,
        name: str = None,
    ) -> None:
        super().__init__()

        self.traj_encoder = traj_encoder
        self.text_encoder = text_encoder
        self.traj_decoder = traj_decoder

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # lambda weighting for the losses
        self.lmd = lmd
        self.lr = lr

        self.name = name

    # --------------------------------------------------------------------------------- #

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    # --------------------------------------------------------------------------------- #

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "traj", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "traj":
            return self.traj_encoder

        m_num_feats = self.traj_encoder.num_feats
        t_num_feats = self.text_encoder.num_feats

        if m_num_feats == t_num_feats:
            raise ValueError(
                "Cannot automatically find the encoder (they share the same input dim)."
            )

        num_feats = inputs["x"].shape[-1]
        if num_feats == m_num_feats:
            return self.traj_encoder
        elif num_feats == t_num_feats:
            return self.text_encoder
        else:
            raise ValueError("The inputs is not recognized.")

    def encode(
        self,
        inputs,
        modality: str = "auto",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs["x"], inputs["mask"])
        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors

    def decode(
        self,
        latent_vectors: Tensor,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
    ):
        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        trajectories = self.traj_decoder(latent_vectors, mask)
        return trajectories

    # Forward: X => trajectories
    def forward(
        self,
        inputs,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:
        # Encoding the inputs and sampling if needed
        latent_vectors, distributions = self.encode(
            inputs, sample_mean=sample_mean, fact=fact, return_distribution=True
        )
        # Decoding the latent vector: generating trajectories
        trajectories = self.decode(latent_vectors, lengths, mask)

        if return_all:
            return trajectories, latent_vectors, distributions

        return trajectories
