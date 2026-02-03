# Paths, columns, seeds, bins

from pathlib import Path

# Paths
def train_data_path() -> Path:
    """
    Returns the location of train data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the train data directory
    """
    cwd = Path.cwd()
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data" / "train.csv"
        if data_folder.exists() and data_folder.is_file():
            print("Train data directory found in ", data_folder)
            return data_folder
    raise Exception("Train data not found")
        
def test_data_path() -> Path:
    """
    Returns the location of test data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the test data directory
    """
    cwd = Path.cwd()
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data" / "test.csv"
        if data_folder.exists() and data_folder.is_file():
            print("Test data directory found in ", data_folder)
            return data_folder
    raise Exception("Test data not found")
        
# Feature columns
COL_EVENT_ID = "event_id"
COL_NUM_PERIMETERS_0_5H = "num_perimeters_0_5h"
COL_DT_FIRST_LAST_0_5H = "dt_first_last_0_5h"
COL_LOW_TEMPORAL_RESOLUTION_0_5H = "low_temporal_resolution_0_5h"
COL_AREA_FIRST_HA = "area_first_ha"
COL_AREA_GROWTH_ABS_0_5H = "area_growth_abs_0_5h"
COL_AREA_GROWTH_REL_0_5H = "area_growth_rel_0_5h"
COL_AREA_GROWTH_RATE_HA_PER_H = "area_growth_rate_ha_per_h"
COL_LOG1P_AREA_FIRST = "log1p_area_first"
COL_LOG1P_GROWTH = "log1p_growth"
COL_LOG_AREA_RATIO_0_5H = "log_area_ratio_0_5h"
COL_RELATIVE_GROWTH_0_5H = "relative_growth_0_5h"
COL_RADIAL_GROWTH_M = "radial_growth_m"
COL_RADIAL_GROWTH_RATE_M_PER_H = "radial_growth_rate_m_per_h"
COL_CENTROID_DISPLACEMENT_M = "centroid_displacement_m"
COL_CENTROID_SPEED_M_PER_H = "centroid_speed_m_per_h"
COL_SPREAD_BEARING_DEG = "spread_bearing_deg"
COL_SPREAD_BEARING_SIN = "spread_bearing_sin"
COL_SPREAD_BEARING_COS = "spread_bearing_cos"
COL_DIST_MIN_CI_0_5H = "dist_min_ci_0_5h"
COL_DIST_STD_CI_0_5H = "dist_std_ci_0_5h"
COL_DIST_CHANGE_CI_0_5H = "dist_change_ci_0_5h"
COL_DIST_SLOPE_CI_0_5H = "dist_slope_ci_0_5h"
COL_CLOSING_SPEED_M_PER_H = "closing_speed_m_per_h"
COL_CLOSING_SPEED_ABS_M_PER_H = "closing_speed_abs_m_per_h"
COL_PROJECTED_ADVANCE_M = "projected_advance_m"
COL_DIST_ACCEL_M_PER_H2 = "dist_accel_m_per_h2"
COL_DIST_FIT_R2_0_5H = "dist_fit_r2_0_5h"
COL_ALIGNMENT_COS = "alignment_cos"
COL_ALIGNMENT_ABS = "alignment_abs"
COL_CROSS_TRACK_COMPONENT = "cross_track_component"
COL_ALONG_TRACK_SPEED = "along_track_speed"
COL_EVENT_START_HOUR = "event_start_hour"
COL_EVENT_START_DAYOFWEEK = "event_start_dayofweek"
COL_EVENT_START_MONTH = "event_start_month"

FEATURE_COLUMNS = [
    COL_NUM_PERIMETERS_0_5H,
    COL_DT_FIRST_LAST_0_5H,
    COL_LOW_TEMPORAL_RESOLUTION_0_5H,
    COL_AREA_FIRST_HA,
    COL_AREA_GROWTH_ABS_0_5H,
    COL_AREA_GROWTH_REL_0_5H,
    COL_AREA_GROWTH_RATE_HA_PER_H,
    COL_LOG1P_AREA_FIRST,
    COL_LOG1P_GROWTH,
    COL_LOG_AREA_RATIO_0_5H,
    COL_RELATIVE_GROWTH_0_5H,
    COL_RADIAL_GROWTH_M,
    COL_RADIAL_GROWTH_RATE_M_PER_H,
    COL_CENTROID_DISPLACEMENT_M,
    COL_CENTROID_SPEED_M_PER_H,
    COL_SPREAD_BEARING_SIN,
    COL_SPREAD_BEARING_COS,
    COL_DIST_MIN_CI_0_5H,
    COL_DIST_STD_CI_0_5H,
    COL_DIST_CHANGE_CI_0_5H,
    COL_DIST_SLOPE_CI_0_5H,
    COL_CLOSING_SPEED_M_PER_H,
    COL_CLOSING_SPEED_ABS_M_PER_H,
    COL_PROJECTED_ADVANCE_M,
    COL_DIST_ACCEL_M_PER_H2,
    COL_DIST_FIT_R2_0_5H,
    COL_ALIGNMENT_COS,
    COL_ALIGNMENT_ABS,
    COL_CROSS_TRACK_COMPONENT,
    COL_ALONG_TRACK_SPEED,
    COL_EVENT_START_HOUR,
    COL_EVENT_START_DAYOFWEEK,
    COL_EVENT_START_MONTH,
]

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