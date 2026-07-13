import pandas as pd
import warnings

    
def count_missing(df, axis=0):
    """
    Count missing values along a specified axis.

    Parameters:
    -----------
    df: pandas DataFrame
        Input dataframe.

    axis: int, default=0
        Axis along which to count missing values.
        0 = columns
        1 = rows

    Returns:
    --------
    pandas Series
        Missing value counts.
    """
    return df.isna().sum(axis=axis)

def shade(series):
    """
    Convert a Series to a shadow variable.
    """
    return series.isna().map({
        True: "Missing",
        False: "Not Missing"
    }).astype("string")