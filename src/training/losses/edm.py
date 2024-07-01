import torch
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

batch_size, num_cams, num_feats, num_condfeats = None, None, None, None
Feats = TensorType["batch_size", "num_feats", "num_cams"]
Labels = TensorType["batch_size", "num_condfeats"]

# ------------------------------------------------------------------------------------- #


class EDMLoss:
    """
    Improved loss function proposed in the paper "Elucidating the Design Space
    of Diffusion-Based Generative Models" (EDM).
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        **kwargs
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        net: torch.nn.Module,
        data: Feats,
        labels: Labels = None,
        mask: Feats = None,
    ) -> Feats:
        rnd_normal = torch.randn([data.shape[0], 1, 1], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(data) * sigma
        D_yn = net(data + n, sigma.squeeze(), y=labels, mask=mask)
        loss = weight * ((D_yn - data) ** 2)

        return loss
