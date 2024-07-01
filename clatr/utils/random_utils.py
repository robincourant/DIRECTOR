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
