import json
import os
import os.path as osp
import pickle
import subprocess
from typing import Any

import pandas as pd
import torch

num_channels, num_frames, height, width = None, None, None, None


def create_dir(dir_name: str):
    """Create a directory if it does not exist yet."""
    if not osp.exists(dir_name):
        os.makedirs(dir_name)


def move_files(source_path: str, destpath: str):
    """Move files from `source_path` to `dest_path`."""
    subprocess.call(["mv", source_path, destpath])


def load_pickle(pickle_path: str) -> Any:
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load_txt(txt_path: str):
    """Load a txt file."""
    with open(txt_path, "r") as f:
        data = f.read()
    return data


def save_txt(data: str, txt_path: str):
    """Save data in a txt file."""
    with open(txt_path, "w") as f:
        f.write(data)


def load_pth(pth_path: str) -> Any:
    """Load a pth (PyTorch) file."""
    data = torch.load(pth_path)
    return data


def save_pth(data: Any, pth_path: str):
    """Save data in a pth (PyTorch) file."""
    torch.save(data, pth_path)


def load_csv(csv_path: str, header: Any = None) -> pd.DataFrame:
    """Load a csv file."""
    try:
        data = pd.read_csv(csv_path, header=header)
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()
    return data


def save_csv(data: Any, csv_path: str):
    """Save data in a csv file."""
    pd.DataFrame(data).to_csv(csv_path, header=False, index=False)


def load_json(json_path: str, header: Any = None) -> pd.DataFrame:
    """Load a json file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Any, json_path: str):
    """Save data in a json file."""
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
