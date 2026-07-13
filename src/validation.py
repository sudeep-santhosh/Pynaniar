import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype

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

def validate_search(search):

    if search is None:
        raise ValueError("'search' cannot be None.")

    if isinstance(search, (list, tuple, set)) and len(search) == 0:
        raise ValueError("'search' cannot be empty.")
    
def validate_column(df, column,check=1):

    if len(column) == 0 and check == 1:
        raise ValueError("Column name cannot be empty.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

def validation_span_check(span):

    if not isinstance(span, int):
        raise TypeError(
            "'span_every' must be an integer."
        )
    
    if not isinstance(span, int) or span <= 0:
        raise ValueError("'span_every' must be a positive integer.")
    

def validation_x(x):
    if x is None:
        raise ValueError("'x' cannot be None.")


def validate_columns(df, columns):
    """
    Validate that all specified columns exist in the DataFrame.
    """

    if columns is None:
        return

    if len(columns) == 0:
        raise ValueError("Columns list cannot be empty.")

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Column(s) not found: {missing_cols}"
        )

def validate_numeric_series(x):
    """
    Validate that x is a numeric pandas Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x must be a pandas Series")

    if not is_numeric_dtype(x):
        raise TypeError("x must be numeric")

def any_shadow(df):
    """
    Check whether a dataframe contains any shadow columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    bool
        True if at least one shadow column exists,
        otherwise False.
    """
    validate_dataframe(df)

    return any(col.endswith("_NA") for col in df.columns)

def validate_series(x, name="x"):
    """
    Validate that the input is a pandas Series.

    Parameters
    ----------
    x : object
        Object to validate.
    name : str, default="x"
        Name of the argument for error messages.

    Raises
    ------
    TypeError
        If x is not a pandas Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError(f"'{name}' must be a pandas Series.")
    
def validate_categorical_series(x, name="x"):
    """
    Validate that the input is a categorical pandas Series.

    Parameters
    ----------
    x : pandas.Series
        Series to validate.
    name : str, default="x"
        Name of the argument for error messages.

    Raises
    ------
    TypeError
        If x is not a pandas Series or does not have a categorical dtype.
    """
    validate_series(x, name)

    if not isinstance(x.dtype, pd.CategoricalDtype):
        raise TypeError(f"'{name}' must have a categorical dtype.")

def validate_string(value, name="value"):
    """
    Validate that the input is a string.

    Parameters
    ----------
    value : object
        Value to validate.
    name : str, default="value"
        Name of the argument for error messages.

    Raises
    ------
    TypeError
        If value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a string.")
