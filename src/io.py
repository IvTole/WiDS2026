import pandas as pd
from typing import Optional, Tuple

# External modules
from src.config import train_data_path, test_data_path
from src.config import COL_EVENT, COL_TIME_TO_HIT_HOURS, COL_EVENT_ID
from src.config import TIME_BINS_HOURS, N_CLASSES, CENSORED_CLASS
from src.targets import make_multiclass_labels


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

class Dataset:

    def __init__(self, num_samples: int = None, random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        """

        self.num_samples = num_samples
        self.random_seed = random_seed

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: Tuple (df_train, df_test)
        """

        train_path = train_data_path()
        test_path = test_data_path()

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Sample
        if self.num_samples is not None:
            df_train = df_train.sample(self.num_samples, random_state=self.random_seed)   

        return df_train, df_test
    
    def load_data_xy(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        :return: Tuple (X_train, y_train, X_test)

        Note: converts (time_to_hit_hours, event) into a 4-class label.
        """

        df_train, df_test = self.load_data()

        # Labels creation, categorical (y_12h, y_24h, y_48h, y_72h)
        y_train = make_multiclass_labels(df=df_train,
                                        bin_edges=TIME_BINS_HOURS,
                                        censored_class=CENSORED_CLASS)
        
        # Drop old target columns
        X_train = df_train.drop(columns=[COL_EVENT, COL_TIME_TO_HIT_HOURS, COL_EVENT_ID], axis=1)
        X_test = df_test.drop(columns=[COL_EVENT_ID], axis=1)

        return X_train, y_train, X_test

