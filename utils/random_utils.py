import numpy as np
import random
import torch


def set_random_seed(seed: int):
    torch.manual_seed((seed) % (1 << 31))
    torch.cuda.manual_seed((seed) % (1 << 31))
    torch.cuda.manual_seed_all((seed) % (1 << 31))
    np.random.seed((seed) % (1 << 31))
    random.seed((seed) % (1 << 31))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed for each
    sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 31)) for seed in seeds
        ]

    def randn_rn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn_rn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )
