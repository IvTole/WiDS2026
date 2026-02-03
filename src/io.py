import pandas as pd
from typing import Optional, Tuple

# External modules
from src.config import train_data_path, test_data_path
from src.config import FEATURE_COLUMNS, COL_EVENT, COL_TIME_TO_HIT_HOURS
from src.config import TIME_BINS_HOURS, N_CLASSES, CENSORED_CLASS
from src.targets import make_multiclass_labels

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
        X_train = df_train.drop([COL_EVENT, COL_TIME_TO_HIT_HOURS], axis=1)
        X_test = df_test

        return X_train, y_train, X_test

