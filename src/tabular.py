import pandas as pd
from utils import count_missing
from validation import validate_dataframe
import numpy as np

def miss_var_summary(df):
    """
    Summarizes missing vlaues for each column in pandas DataFrame.

    Parameters:
    -----------
    df: pandas Dataframe 
    
    Returns:
    --------
    pandas dataframe 
    Summary table with missingness statistics per variable
    """

    validate_dataframe(df)
    
    summary = pd.DataFrame({
        "variable": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "n_miss": df.isna().sum().values,
        "pct_miss": (df.isna().mean() * 100).round(2).values,
        "n_complete": df.notna().sum().values
    })

    return summary.sort_values(
        by="pct_miss",
        ascending=False
    ).reset_index(drop=True)

def miss_var_table(df):
    """
    Create a frequency table of missing values for each variable in a pandas DataFrame.

    Parameters:
    -----------
    df: pandas DataFrame

    Returns:
    --------
    pandas DataFrame
        A table showing:
        - n_miss_in_var: number of missing values in a variable
        - n_vars: number of variables with that missing count
        - pct_vars: percentage of total variables
    """
    validate_dataframe(df)
    miss_counts = count_missing(df, axis=0)
    table=(
        miss_counts.value_counts().reset_index()
    )
    table.columns=["n_miss_in_var","n_vars"]

    table["pct_vars"] = (
        table["n_vars"] / len(df.columns) * 100
    ).round(2)

    return table.sort_values(
        by="n_miss_in_var",
        ascending=False
    ).reset_index(drop=True)   


def miss_case_table(df):
    """
    Create a frequency table of missing values for each case (row) in a pandas DataFrame.

    Parameters:
    -----------
    df: pandas DataFrame

    Returns:
    pandas DataFrame
        A table showing:
        - n_miss_in_case: number of missing values in a row 
        - n_cases: number of cases with that missing count
        - pct_cases: percentage of total cases
    """

    validate_dataframe(df)

    miss_counts = count_missing(df, axis=1)
    table=(
        miss_counts.value_counts().reset_index()
    )
    table.columns=["n_miss_in_case","n_cases"]

    table["pct_cases"]=(
        table["n_cases"]/len(df)*100
    ).round(2)

    return table.sort_values(by="n_miss_in_case", ascending=False).reset_index(drop=True)

def miss_case_summary(df):
    """
    Create a summary of missing values for each case (row) in a pandas DataFrame.

    Parameters:
    -----------
    df: pandas DataFrame
        Input dataframe to analyze.

    Returns:
    --------
    pandas DataFrame
        A summary table showing:
        - case: row identifier / index
        - n_miss: number of missing values in the row
        - pct_miss: percentage of missing values in the row
        - n_complete: number of non-missing values in the row
    """
    validate_dataframe(df)
    summary=pd.DataFrame(
        {
            "case": df.index,
            "n_miss": count_missing(df, axis=1),
            "pct_miss":(df.isna().mean(axis=1) * 100).round(2).values,
            "n_complete": df.notna().sum(axis=1).values
        }
    )
    return summary.sort_values(
        by="pct_miss",
        ascending=False
    ).reset_index(drop=True)


def add_shadow(
    df,
    suffix="_NA",
    missing_label="NA",
    complete_label="!NA",
    only_missing=False
):
    """
    Create shadow columns indicating missingness.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    suffix : str, default="_NA"
        Suffix to append to shadow columns.

    missing_label : str, default="NA"
        Label used for missing values.

    complete_label : str, default="!NA"
        Label used for non-missing values.

    only_missing : bool, default=False
        If True, return only shadow columns.
        If False, bind shadow columns to original dataframe.

    Returns
    -------
    pandas.DataFrame
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    shadow = df.isna().replace({
        True: missing_label,
        False: complete_label
    })

    shadow.columns = [f"{col}{suffix}" for col in df.columns]

    if only_missing:
        return shadow

    return pd.concat([df, shadow], axis=1)


def miss_case_cumsum(df):
    """
    Calculate cumulative sum of missing values across cases (rows).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to analyze.

    Returns
    -------
    pandas.DataFrame
        A table showing:
        - case: row identifier / index
        - n_miss: number of missing values in the row
        - n_miss_cumsum: cumulative sum of missing values
    """
    validate_dataframe(df)

    result = pd.DataFrame({
        "case": df.index,
        "n_miss": df.isna().sum(axis=1)
    })

    result = result.sort_values(
        by="n_miss",
        ascending=False
    ).reset_index(drop=True)

    result["n_miss_cumsum"] = result["n_miss"].cumsum()

    return result



