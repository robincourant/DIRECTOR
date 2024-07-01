from typing import Any, Dict

import pandas as pd

METRICS_TO_KEEP = [
    "rot_rate/rot_rate",
    "ftd/ftd",
    "rot_kl/rot_kl",
    "trans_kl/trans_kl",
    "rot_emd/rot_emd",
    "trans_emd/trans_emd",
    "rot_prdc/precision",
    "rot_prdc/recall",
    "rot_prdc/density",
    "rot_prdc/coverage",
    "trans_prdc/precision",
    "trans_prdc/recall",
    "trans_prdc/density",
    "trans_prdc/coverage",
]


def to_dataframe(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    indices, values = [], []
    for key, value in metrics_dict.items():
        if key not in METRICS_TO_KEEP:
            continue
        group, metric = key.split("/")
        if "rot" in group:
            indices.append(("rot", metric))
        elif "trans" in group:
            indices.append(("trans", metric))
        else:
            indices.append(("-", metric))
        values.append(value)
    metrics_df = pd.DataFrame(
        pd.Series(values, index=pd.MultiIndex.from_tuples(indices)).sort_index()
    ).T

    return metrics_df
