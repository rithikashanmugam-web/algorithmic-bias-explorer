import pandas as pd
import numpy as np
from config import DATA_PATH, COLUMNS, TARGET_COLUMN, TARGET_MAP


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        names=COLUMNS,
        sep=", ",
        engine="python"
    )

    print("\nMissing values:")
    print((df == "?").sum())

    df.replace("?", np.nan, inplace=True)
    df = df.dropna().copy()

    # Convert target if mapping provided
    if TARGET_MAP:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    return df
