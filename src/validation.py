import warnings
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
    
    if df.shape[0] == 0:
        raise ValueError("DataFrame must contain at least one row.")

    if df.shape[1] == 0:
        raise ValueError("DataFrame must contain at least one column.")
    
    if df.columns.duplicated().any():
        warnings.warn("DataFrame contains duplicate column names.")

