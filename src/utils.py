import pandas as pd


def validate_dataframe(df):
    """
    Validates that the input is a pandas DataFrame.

    Parameters:
    -----------
    df: any 
        The input to validate.

    Raises:
    -------
    ValueError: If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    

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
    validate_dataframe(df)
    return df.isna().sum(axis=axis)