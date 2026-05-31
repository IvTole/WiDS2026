# Paths, columns, seeds, bins

from pathlib import Path

# mlflow issues
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.vanotole-lab.com"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

# Paths
def _find_data_file(filename: str) -> Path:
    cwd = Path.cwd()
    for folder in (cwd, cwd / '..', cwd / '..' / '..'):
        path = folder /'data' / filename
        if path.exits():
            return path
    raise FileNotFoundError(f'{filename} not found')

def train_data_path() -> Path: return _find_data_file('train.csv')
def test_data_path() -> Path: return _find_data_file('test.csv')

# Feature columns
COL_EVENT_ID = "event_id"

# Target columns
COL_TIME_TO_HIT_HOURS = "time_to_hit_hours"
COL_EVENT = "event"

# Target parameters
TIME_BINS_HOURS = [0, 12, 24, 48, 72]   # ejemplo
N_CLASSES = 4
CENSORED_CLASS = 3

# MLFlow
MLFLOW_TRACKING_URL = "http://mlflow.vanotole-lab.com"
MLFLOW_EXPERIMENT_NAME = "WiDS2026"

FEATURE_COLUMNS = [
    "num_perimeters_0_5h",
    "dt_first_last_0_5h",
    "low_temporal_resolution_0_5h",
    "area_first_ha",
    "area_growth_abs_0_5h",
    "area_growth_rel_0_5h",
    "area_growth_rate_ha_per_h",
    "log1p_area_first",
    "log1p_growth",
    "log_area_ratio_0_5h",
    "relative_growth_0_5h",
    "radial_growth_m",
    "radial_growth_rate_m_per_h",
    "centroid_displacement_m",
    "centroid_speed_m_per_h",
    "spread_bearing_sin",
    "spread_bearing_cos",
    "dist_min_ci_0_5h",
    "dist_std_ci_0_5h",
    "dist_change_ci_0_5h",
    "dist_slope_ci_0_5h",
    "closing_speed_m_per_h",
    "closing_speed_abs_m_per_h",
    "projected_advance_m",
    "dist_accel_m_per_h2",
    "dist_fit_r2_0_5h",
    "alignment_cos",
    "alignment_abs",
    "cross_track_component",
    "along_track_speed",
    "event_start_hour",
    "event_start_dayofweek",
    "event_start_month"
]