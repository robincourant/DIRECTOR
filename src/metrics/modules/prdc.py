"""Code adapted from: https://github.com/clovaai/generative-evaluation-prdc"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import scipy

from utils.rotation_utils import pairwise_geodesic


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class ManifoldMetrics(Metric):
    def __init__(
        self,
        reset_real_features: bool = True,
        manifold_k: int = 3,
        distance: str = "geodesic",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.manifold_k = manifold_k
        self.reset_real_features = reset_real_features
        self.distance = distance

        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")

    # --------------------------------------------------------------------------------- #

    def _compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = torch.clone(data_x)

        if self.distance == "euclidean":
            num_feats = data_x.shape[-1]
            X = data_x.reshape(-1, num_feats).unsqueeze(0)
            Y = data_y.reshape(-1, num_feats).unsqueeze(0)
            dists = torch.cdist(X, Y, 2).squeeze(0)

        if self.distance == "geodesic":
            dists = pairwise_geodesic(data_x, data_y)

        return dists

    def _get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        # indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        # k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        # kth_values = k_smallests.max(axis=axis)

        k_smallests = torch.topk(unsorted, k, largest=False, dim=-1)
        kth_values = k_smallests.values.max(axis=axis).values
        return kth_values

    def _compute_nn_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def compute_prdc(self, real_features, fake_features, nearest_k):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_nearest_neighbour_distances = self._compute_nn_distances(
            real_features, nearest_k
        )
        fake_nearest_neighbour_distances = self._compute_nn_distances(
            fake_features, nearest_k
        )
        distance_real_fake = self._compute_pairwise_distance(
            real_features, fake_features
        )

        precision = (
            (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1))
            .any(axis=0)
            .to(float)
        ).mean()

        recall = (
            (distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(1))
            .any(axis=1)
            .to(float)
            .mean()
        )

        density = (1.0 / float(nearest_k)) * (
            distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        ).sum(axis=0).to(float).mean()

        coverage = (
            (distance_real_fake.min(axis=1).values < real_nearest_neighbour_distances)
            .to(float)
            .mean()
        )
        return precision, recall, density, coverage

    # --------------------------------------------------------------------------------- #

    def update(self, real_features, fake_features):
        """Updates the state with new real and fake features."""
        self.real_features.append(real_features)
        self.fake_features.append(fake_features)

    def compute(self, num_splits=5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int.
            num_splits: int. Number of splits to use for computing metrics.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_features = dim_zero_cat(self.real_features).chunk(num_splits, dim=0)
        fake_features = dim_zero_cat(self.fake_features).chunk(num_splits, dim=0)
        precision, recall, density, coverage = [], [], [], []
        for real, fake in zip(real_features, fake_features):
            p, r, d, c = self.compute_prdc(real, fake, nearest_k=self.manifold_k)
            precision.append(p)
            recall.append(r)
            density.append(d)
            coverage.append(c)

        precision = torch.stack(precision).mean()
        recall = torch.stack(recall).mean()
        density = torch.stack(density).mean()
        coverage = torch.stack(coverage).mean()

        return precision, recall, density, coverage
