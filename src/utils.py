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
    

