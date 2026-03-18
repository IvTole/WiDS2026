from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.config import NUMERIC_COLS, PASSTHROUGH_COLS

def build_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERIC_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS)
        ]
    )

    return preprocessor