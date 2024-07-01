import torch.nn as nn


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(nn.Module):
    def __init__(self, sigma_data: float = 0.5, module: nn.Module = None, **kwargs):
        super().__init__()
        self.sigma_data = sigma_data

        self.model = module
        self.num_rawfeats = module.num_rawfeats
        self.num_feats = module.num_feats
        self.num_cams = module.num_cams

    def forward(self, x, sigma, y=None, mask=None):
        """
        x: [batch_size, num_feats, max_frames], denoted x_t in the paper
        sigma: [batch_size] (int)
        """
        sigma = sigma.reshape(-1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, c_noise.flatten(), y=y, mask=mask)
        D_x = c_skip * x + c_out * F_x

        return D_x
