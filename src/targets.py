# src/targets.py
import numpy as np
import pandas as pd
from src.config import COL_TIME_TO_HIT_HOURS, COL_EVENT

def make_multiclass_labels(
    df: pd.DataFrame,
    bin_edges: list[float],
    censored_class: int | None = None,
) -> np.ndarray:
    """
    Converts (time_to_hit_hours, event) into a 4-class label.

    bin_edges: e.g. [0, 12, 24, 48, 72] -> 4 bins
    censored_class: if None, uses last bin index (len(bin_edges)-2)
    """
    t = df[COL_TIME_TO_HIT_HOURS].to_numpy()
    e = df[COL_EVENT].to_numpy().astype(int)

    n_bins = len(bin_edges) - 1
    if censored_class is None:
        censored_class = n_bins - 1

    # Bin assignment for observed events
    y = np.digitize(t, bin_edges[1:-1], right=False)

    # For censored events -> forced to censored_class
    y[e == 0] = censored_class

    return y
