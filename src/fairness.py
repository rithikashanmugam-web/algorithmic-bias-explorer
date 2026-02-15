import pandas as pd


def _convert_to_numeric(series):
    """
    Converts labels/predictions to numeric automatically.
    Works for any dataset.
    """
    # already numeric → keep
    if pd.api.types.is_numeric_dtype(series):
        return series

    # convert categories/strings → numbers
    return pd.factorize(series)[0]


def demographic_parity(y_true, y_pred, sensitive_feature):
    """
    Computes prediction rate per group.
    Works for any datatype.
    """

    y_pred = _convert_to_numeric(pd.Series(y_pred))

    df = pd.DataFrame({
        "group": sensitive_feature,
        "pred": y_pred
    })

    return df.groupby("group")["pred"].mean()


def equal_opportunity(y_true, y_pred, sensitive_feature):
    """
    True positive rate per group.
    """

    y_true = _convert_to_numeric(pd.Series(y_true))
    y_pred = _convert_to_numeric(pd.Series(y_pred))

    df = pd.DataFrame({
        "group": sensitive_feature,
        "true": y_true,
        "pred": y_pred
    })

    df = df[df["true"] == 1]

    return df.groupby("group")["pred"].mean()
