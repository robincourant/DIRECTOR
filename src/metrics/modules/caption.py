from itertools import product
from typing import List, Tuple

from evo.core import lie_algebra as lie
import numpy as np
import torch
from scipy.stats import mode
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torchmetrics.functional as F
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_samples, num_cams, num_total_cams, num_classes = None, None, None, None
width, height = None, None

# ------------------------------------------------------------------------------------- #

CAM_INDEX_TO_PATTERN = {
    0: "static",
    1: "push_in",
    2: "pull_out",
    3: "boom_bottom",
    6: "boom_top",
    18: "trucking_left",
    9: "trucking_right",
    # ----- #
    12: "trucking_right-boom_bottom",
    15: "trucking_right-boom_top",
    21: "trucking_left-boom_bottom",
    24: "trucking_left-boom_top",
    10: "trucking_right-push_in",
    11: "trucking_right-pull_out",
    19: "trucking_left-push_in",
    20: "trucking_left-pull_out",
    4: "boom_bottom-push_in",
    5: "boom_bottom-pull_out",
    7: "boom_top-push_in",
    8: "boom_top-pull_out",
    # ----- #
    13: "trucking_right-boom_bottom-push_in",
    14: "trucking_right-boom_bottom-pull_out",
    16: "trucking_right-boom_top-push_in",
    17: "trucking_right-boom_top-pull_out",
    22: "trucking_left-boom_bottom-push_in",
    23: "trucking_left-boom_bottom-pull_out",
    25: "trucking_left-boom_top-push_in",
    26: "trucking_left-boom_top-pull_out",
}

# ------------------------------------------------------------------------------------- #


def to_euler_angles(
    rotation_mat: TensorType["num_samples", 3, 3]
) -> TensorType["num_samples", 3]:
    rotation_vec = torch.from_numpy(
        np.stack(
            [lie.sst_rotation_from_matrix(r).as_rotvec() for r in rotation_mat.numpy()]
        )
    )
    return rotation_vec


def compute_relative(f_t: TensorType["num_samples", 3]):
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 1])]), axis=0)
    xy_f_t = np.divide(
        (abs(f_t[:, 0]) - abs(f_t[:, 1])),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 2])]), axis=0)
    xz_f_t = np.divide(
        abs(f_t[:, 0]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 1]), abs(f_t[:, 2])]), axis=0)
    yz_f_t = np.divide(
        abs(f_t[:, 1]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    return xy_f_t, xz_f_t, yz_f_t


def compute_camera_dynamics(w2c_poses: TensorType["num_samples", 4, 4], fps: float):
    w2c_poses_inv = torch.from_numpy(
        np.array([lie.se3_inverse(t) for t in w2c_poses.numpy()])
    )
    velocities = w2c_poses_inv[:-1].to(float) @ w2c_poses[1:].to(float)

    # --------------------------------------------------------------------------------- #
    # Translation velocity
    t_velocities = fps * velocities[:, :3, 3]
    t_xy_velocity, t_xz_velocity, t_yz_velocity = compute_relative(t_velocities)
    t_vels = (t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity)
    # --------------------------------------------------------------------------------- #
    # # Rotation velocity
    # a_velocities = to_euler_angles(velocities[:, :3, :3])
    # a_xy_velocity, a_xz_velocity, a_yz_velocity = compute_relative(a_velocities)
    # a_vels = (a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity)

    return velocities, t_vels, None


# ------------------------------------------------------------------------------------- #


def perform_segmentation(
    velocities: TensorType["num_samples-1", 3],
    xy_velocity: TensorType["num_samples-1", 3],
    xz_velocity: TensorType["num_samples-1", 3],
    yz_velocity: TensorType["num_samples-1", 3],
    static_threshold: float,
    diff_threshold: float,
) -> List[int]:
    segments = torch.zeros(velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    for sample_index, sample_velocity in enumerate(velocities):
        sample_pattern = abs(sample_velocity) > static_threshold

        # XY
        if (sample_pattern == torch.tensor([1, 1, 0])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])

        # XZ
        elif (sample_pattern == torch.tensor([1, 0, 1])).all():
            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # YZ
        elif (sample_pattern == torch.tensor([0, 1, 1])).all():
            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # XYZ
        elif (sample_pattern == torch.tensor([1, 1, 1])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern[1] = 0
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern[1] = 0

        sample_pattern = torch.sign(sample_velocity) * sample_pattern
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]

    return np.array(segments, dtype=int)


def smooth_segments(arr: List[int], window_size: int) -> List[int]:
    smoothed_arr = arr.copy()

    if len(arr) < window_size:
        return smoothed_arr

    half_window = window_size // 2
    # Handle the first half_window elements
    for i in range(half_window):
        window = arr[: i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    for i in range(half_window, len(arr) - half_window):
        window = arr[i - half_window : i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    # Handle the last half_window elements
    for i in range(len(arr) - half_window, len(arr)):
        window = arr[i - half_window :]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    return smoothed_arr


def remove_short_chunks(arr: List[int], min_chunk_size: int) -> List[int]:
    def remove_chunk(chunks):
        if len(chunks) == 1:
            return False, chunks

        chunk_lenghts = [(end - start) + 1 for _, start, end in chunks]
        chunk_index = np.argmin(chunk_lenghts)
        chunk_length = chunk_lenghts[chunk_index]
        if chunk_length < min_chunk_size:
            _, start, end = chunks[chunk_index]

            # Check if the chunk is at the beginning
            if chunk_index == 0:
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index + 1] = (segment_r, start_r - chunk_length, end_r)

            elif chunk_index == len(chunks) - 1:
                segment_l, start_l, end_l = chunks[chunk_index - 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + chunk_length)

            else:
                if chunk_length % 2 == 0:
                    half_length_l = chunk_length // 2
                    half_length_r = chunk_length // 2
                else:
                    half_length_l = (chunk_length // 2) + 1
                    half_length_r = chunk_length // 2

                segment_l, start_l, end_l = chunks[chunk_index - 1]
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + half_length_l)
                chunks[chunk_index + 1] = (segment_r, start_r - half_length_r, end_r)

            chunks.pop(chunk_index)

        return chunk_length < min_chunk_size, chunks

    chunks = find_consecutive_chunks(arr)
    keep_removing, chunks = remove_chunk(chunks)
    while keep_removing:
        keep_removing, chunks = remove_chunk(chunks)

    merged_chunks = []
    for segment, start, end in chunks:
        merged_chunks.extend([segment] * ((end - start) + 1))

    return merged_chunks


# ------------------------------------------------------------------------------------- #


def find_consecutive_chunks(arr: List[int]) -> List[Tuple[int, int, int]]:
    chunks = []
    start_index = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            end_index = i - 1
            if end_index >= start_index:
                chunks.append((arr[start_index], start_index, end_index))
            start_index = i

    # Add the last chunk if the array ends with consecutive similar digits
    if start_index < len(arr):
        chunks.append((arr[start_index], start_index, len(arr) - 1))

    return chunks


# ------------------------------------------------------------------------------------- #


class CaptionMetrics(Metric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.metric_kwargs = dict(
            task="multiclass",
            num_classes=num_classes,
            average="weighted",
            zero_division=0,
        )

        self.fps = 25.0
        self.cam_static_threshold = 0.02
        self.cam_diff_threshold = 0.4
        self.smoothing_window_size = 56
        self.min_chunk_size = 25

        self.add_state("pred_segments", default=[], dist_reduce_fx="cat")
        self.add_state("target_segments", default=[], dist_reduce_fx="cat")

    def segment_camera_trajectories(
        self, w2c_poses: TensorType["num_samples", 4, 4]
    ) -> TensorType["num_samples"]:
        device = w2c_poses.device

        _, t_vels, _ = compute_camera_dynamics(w2c_poses.cpu(), fps=self.fps)
        cam_velocities, cam_xy_velocity, cam_xz_velocity, cam_yz_velocity = t_vels

        # Translation segments
        cam_segments = perform_segmentation(
            cam_velocities,
            cam_xy_velocity,
            cam_xz_velocity,
            cam_yz_velocity,
            static_threshold=self.cam_static_threshold,
            diff_threshold=self.cam_diff_threshold,
        )
        cam_segments = smooth_segments(cam_segments, self.smoothing_window_size)
        cam_segments = remove_short_chunks(cam_segments, self.min_chunk_size)
        cam_segments = torch.tensor(cam_segments, device=device)

        return cam_segments

    # --------------------------------------------------------------------------------- #

    def update(
        self,
        trajectories: TensorType["num_samples", "num_cams", 4, 4],
        raw_labels: TensorType["num_samples", "num_classes*num_total_cams"],
        mask: TensorType["num_samples", "num_cams"],
    ) -> Tuple[float, float, float]:
        """Update the state with extracted features."""
        for sample_index in range(trajectories.shape[0]):
            trajectory = trajectories[sample_index][mask[sample_index].to(bool)]
            labels = raw_labels[sample_index][mask[sample_index].to(bool)][:-1]
            if trajectory.shape[0] < 2:
                continue

            self.pred_segments.append(self.segment_camera_trajectories(trajectory))
            self.target_segments.append(labels)

    def compute(self) -> Tuple[float, float, float]:
        """ """
        target_segments = dim_zero_cat(self.target_segments)
        pred_segments = dim_zero_cat(self.pred_segments)

        precision = F.precision(pred_segments, target_segments, **self.metric_kwargs)
        recall = F.recall(pred_segments, target_segments, **self.metric_kwargs)
        fscore = F.f1_score(pred_segments, target_segments, **self.metric_kwargs)

        return precision, recall, fscore
