import math

import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/pytorch/pytorch/issues/50334
    One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])  # slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    indices = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    line_idx = torch.arange(len(indices), device=indices.device).view(-1, 1)
    line_idx = line_idx.expand(indices.shape)
    # idx = torch.cat([line_idx, indices] , 0)
    return m[line_idx, indices].mul(x) + b[line_idx, indices]
