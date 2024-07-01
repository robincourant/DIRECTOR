"""
This code is adapted from https://github.com/Mathux/TMR
"""

from typing import Dict, Optional
from typing import List

from evo.core.trajectory import PosePath3D
from evo.tools.plot import PlotCollection
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchtyping import TensorType
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import clone

from clatr.src.models.temos import TEMOS
from clatr.src.training.losses import InfoNCE_with_filtering
from clatr.src.training.metrics import all_contrastive_metrics
from clatr.utils.visualization import draw_trajectories

# ------------------------------------------------------------------------------------- #

batch_size, num_samples = None, None
num_feats, num_rawfeats, num_cams = None, None, None

# ------------------------------------------------------------------------------------- #


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


# ------------------------------------------------------------------------------------- #


class CLaTr(TEMOS):
    r"""
    Code highly adapated from:
    TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    :param traj_encoder: a module to encode the input traj features in the latent space.
    :param text_encoder: a module to encode the text embeddings in the latent space.
    :param traj_decoder: a module to decode the latent vector into traj features.
    :param vae: a boolean to make the model probabilistic.
    :param fact: a scaling factor for sampling the VAE.
    :param sample_mean: sample the mean vector instead of random sampling.
    :param lmd: dictionary of losses weights.
    :param lr: learninig rate for the optimizer.
    :param temperature: temperature of the softmax in the contrastive loss.
    :param threshold_selfsim: threshold to filter wrong neg for the contrastive loss.
    :param threshold_selfsim_metrics: threshold used to filter wrong neg for the metrics.
    """

    def __init__(
        self,
        traj_encoder: nn.Module,
        text_encoder: nn.Module,
        traj_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
        lr: float = 1e-4,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
        log_wandb: bool = False,
        name: str = None,
        checkpoint_path: str = None,
        device: str = "cuda",
    ) -> None:
        # Initialize module like TEMOS
        super().__init__(
            traj_encoder=traj_encoder,
            text_encoder=text_encoder,
            traj_decoder=traj_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
            name=name,
        )

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_t_trajectories = []
        self.validation_step_m_trajectories = []
        self.validation_step_ref_trajectories = []
        self.validation_step_masks = []
        self.validation_step_sent_token = []
        self.validation_step_sent = []

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location=torch.device(device))[
                "state_dict"
            ]
            print(self.load_state_dict(state_dict))
            for param in self.parameters():
                param.requires_grad = False

    # --------------------------------------------------------------------------------- #

    def on_fit_start(self):
        self.get_matrix = self.trainer.datamodule.eval_dataset.get_matrix

    def on_train_start(self):
        # ----------------------------------------------------------------------------- #
        # https://stackoverflow.com/questions/73095460/assertionerror-if-capturable-false-state-steps-should-not-be-cuda-tensors # noqa
        if self.trainer.ckpt_path is not None and isinstance(
            self.optimizers().optimizer, torch.optim.AdamW
        ):
            self.optimizers().param_groups[0]["capturable"] = True
        # ----------------------------------------------------------------------------- #

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        ref_traj = batch["traj_feat"]
        ref_feats = ref_traj.clone()

        m_inputs = {"x": ref_feats, "mask": batch["padding_mask"]}

        sent_token = batch["caption_raw"]["token"]
        t_inputs = batch["caption_feat"]

        mask = batch["padding_mask"]

        # text -> traj
        t_trajectories, t_latents, t_dists = self(t_inputs, mask=mask, return_all=True)
        # traj -> traj
        m_trajectories, m_latents, m_dists = self(m_inputs, mask=mask, return_all=True)

        # Store all losses
        losses = {}
        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_trajectories, ref_feats)  # text -> traj
            + self.reconstruction_loss_fn(m_trajectories, ref_feats)  # traj -> traj
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(t_dists, m_dists)  # text_to_traj
                + self.kl_loss_fn(m_dists, t_dists)  # traj_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # traj
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent"] = self.latent_loss_fn(t_latents, m_latents)

        # TMR: adding the contrastive loss
        losses["contrastive"] = self.contrastive_loss_fn(
            t_latents, m_latents, sent_token
        )

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Used for the validation step
        if return_all:
            return losses, t_latents, m_latents, t_trajectories, m_trajectories

        return losses

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["traj_feat"])
        losses = self.compute_loss(batch)

        if self.log_wandb:
            for loss_name in sorted(losses):
                loss_val = losses[loss_name]
                self.log(
                    f"train/{loss_name}", loss_val.item(), on_step=True, batch_size=bs
                )

        return losses["loss"]

    # --------------------------------------------------------------------------------- #

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["traj_feat"])
        out = self.compute_loss(batch, return_all=True)
        losses, t_latents, m_latents, t_trajectories, m_trajectories = out

        # Store validation values
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_t_trajectories.append(t_trajectories)
        self.validation_step_m_trajectories.append(m_trajectories)
        self.validation_step_ref_trajectories.append(batch["traj_feat"])
        self.validation_step_masks.append(batch["padding_mask"])
        self.validation_step_sent_token.append(batch["caption_raw"]["token"])
        self.validation_step_sent.append(batch["caption_raw"]["caption"])

        if self.log_wandb:
            for loss_name in sorted(losses):
                loss_val = losses[loss_name]
                self.log(f"val/{loss_name}", loss_val, on_step=True, batch_size=bs)

        return losses["loss"]

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        ref_trajectories = torch.cat(self.validation_step_ref_trajectories)
        t_trajectories = torch.cat(self.validation_step_t_trajectories)
        m_trajectories = torch.cat(self.validation_step_m_trajectories)
        sent_tokens = torch.cat(self.validation_step_sent_token)
        sent = sum(self.validation_step_sent, [])
        masks = torch.cat(self.validation_step_masks)

        # Compute the similarity matrix
        sim_matrix = get_sim_matrix(t_latents, m_latents).cpu().numpy()

        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_tokens.cpu().numpy(),
            threshold=self.threshold_selfsim_metrics,
        )

        if self.log_wandb:
            for loss_name in sorted(contrastive_metrics):
                loss_val = contrastive_metrics[loss_name]
                self.log(f"val/{loss_name}", loss_val)

            ref_matrices = torch.stack(
                [self.get_matrix(x.permute(1, 0)) for x in ref_trajectories[:, :, :9]]
            )
            t_matrices = torch.stack(
                [self.get_matrix(x.permute(1, 0)) for x in t_trajectories[:, :, :9]]
            )
            m_matrices = torch.stack(
                [self.get_matrix(x.permute(1, 0)) for x in m_trajectories[:, :, :9]]
            )

            # Draw reconstructed trajectories
            traj_plots = self.draw_traj_plots(
                ref_matrices=ref_matrices[0].unsqueeze(0).cpu(),
                text_matrices=t_matrices[0].unsqueeze(0).cpu(),
                traj_matrices=m_matrices[0].unsqueeze(0).cpu(),
                masks=masks[0].unsqueeze(0).cpu(),
            )
            latent_plots = self.draw_latent_plots(t_latents, m_latents)
            plot_dict = {**traj_plots[0].figures, **latent_plots}
            for plot_name, raw_plot in plot_dict.items():
                raw_plot.canvas.draw_idle()
                plot_data = np.frombuffer(
                    raw_plot.canvas.tostring_rgb(), dtype=np.uint8
                )
                plot_data = plot_data.reshape(
                    raw_plot.canvas.get_width_height()[::-1] + (3,)
                )
                caption = f"[Epoch {self.current_epoch}]"
                caption += (
                    f" {sent[0]}"
                    if not (("tsne" in plot_name) or ("pca" in plot_name))
                    else ""
                )
                self.logger.log_image(
                    key=f"val/plots/{plot_name}",
                    images=[plot_data],
                    step=self.global_step,
                    caption=[caption],
                )
            plt.close()

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_t_trajectories.clear()
        self.validation_step_m_trajectories.clear()
        self.validation_step_ref_trajectories.clear()
        self.validation_step_masks.clear()
        self.validation_step_sent_token.clear()
        self.validation_step_sent.clear()

    # ---------------------------------------------------------------------------------- #

    def on_predict_start(self):
        self.get_matrix = self.trainer.datamodule.eval_dataset.get_matrix

    def predict_step(self, batch: Dict, batch_idx: int) -> Tensor:
        out = self.compute_loss(batch, return_all=True)
        _, t_latents, m_latents, t_trajectories, m_trajectories = out

        batch["t_latents"] = t_latents
        batch["m_latents"] = m_latents
        batch["t_trajectories"] = t_trajectories
        batch["m_trajectories"] = m_trajectories

        batch["ref_matrices"] = torch.stack(
            [self.get_matrix(x.permute(1, 0)) for x in batch["traj_feat"][:, :, :9]]
        )
        batch["t_matrices"] = torch.stack(
            [self.get_matrix(x.permute(1, 0)) for x in t_trajectories[:, :, :9]]
        )
        batch["m_matrices"] = torch.stack(
            [self.get_matrix(x.permute(1, 0)) for x in m_trajectories[:, :, :9]]
        )

        return batch

    # ---------------------------------------------------------------------------------- #

    @staticmethod
    def draw_traj_plots(
        ref_matrices: TensorType["batch_size", "num_cams", 4, 4],
        text_matrices: TensorType["batch_size", "num_cams", 4, 4],
        traj_matrices: TensorType["batch_size", "num_cams", 4, 4],
        masks: TensorType["batch_size", "num_cams"],
    ) -> List[PlotCollection]:
        plots, colormaps = [], []
        for index in range(len(text_matrices)):
            sample_dict = {}
            mask = masks[index].to(bool)

            ref_se3 = [x for x in ref_matrices[index][mask].numpy()]
            sample_dict["ref_sample"] = PosePath3D(poses_se3=ref_se3)
            colormaps.append("Greens")

            t_se3 = [x for x in text_matrices[index][mask].numpy()]
            sample_dict["t_sample"] = PosePath3D(poses_se3=t_se3)
            colormaps.append("Blues")

            m_se3 = [x for x in traj_matrices[index][mask].numpy()]
            sample_dict["m_sample"] = PosePath3D(poses_se3=m_se3)
            colormaps.append("Reds")

            plot = draw_trajectories(sample_dict, colormaps=colormaps)
            plots.append(plot)

        return plots

    @staticmethod
    def draw_latent_plots(
        t_latents: TensorType["batch_size", "num_feats"],
        m_latents: TensorType["batch_size", "num_feats"],
    ) -> List[PlotCollection]:
        # Perform k-means clustering on the text latents
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
        kmeans.fit(t_latents.cpu().numpy())
        cluster_labels = kmeans.labels_

        # Compute t-SNE embeddings
        tsne = TSNE(n_components=2)
        tsne_t_embeddings = clone(tsne).fit_transform(t_latents.cpu().numpy())
        tsne_m_embeddings = clone(tsne).fit_transform(m_latents.cpu().numpy())

        # Compute PCA embeddings
        pca = PCA(n_components=2)
        pca_t_embeddings = pca.fit_transform(t_latents.cpu().numpy())
        pca_m_embeddings = pca.fit_transform(m_latents.cpu().numpy())

        # Visualize the embeddings with cluster labels
        tsne_fig, tsne_ax = plt.subplots(1, 2)
        pca_fig, pca_ax = plt.subplots(1, 2)
        for i in range(num_clusters):
            # t-SNE visualization
            tsne_t_cluster = tsne_t_embeddings[cluster_labels == i]
            tsne_m_cluster = tsne_m_embeddings[cluster_labels == i]
            tsne_ax[0].scatter(
                tsne_t_cluster[:, 0], tsne_t_cluster[:, 1], label=f"Cluster {i+1}", s=5
            )
            tsne_ax[1].scatter(
                tsne_m_cluster[:, 0], tsne_m_cluster[:, 1], label=f"Cluster {i+1}", s=5
            )
            # PCA visualization
            pca_t_cluster = pca_t_embeddings[cluster_labels == i]
            pca_m_cluster = pca_m_embeddings[cluster_labels == i]
            pca_ax[0].scatter(
                pca_t_cluster[:, 0], pca_t_cluster[:, 1], label=f"Cluster {i+1}", s=5
            )
            pca_ax[1].scatter(
                pca_m_cluster[:, 0], pca_m_cluster[:, 1], label=f"Cluster {i+1}", s=5
            )
        tsne_ax[0].set_title("Text t-SNE Embeddings")
        tsne_ax[1].set_title("Traj t-SNE Embeddings")
        pca_ax[0].set_title("Text PCA Embeddings")
        pca_ax[1].set_title("Traj PCA Embeddings")

        return {"tsne": tsne_fig, "pca": pca_fig}
