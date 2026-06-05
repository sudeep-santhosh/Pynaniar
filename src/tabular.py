import pandas as pd
from utils import count_missing
from validation import validate_dataframe, validate_search, validate_column, validation_span_check,validation_x,validate_columns
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


def miss_prop_summary(df):
    """
    Summarize missing value proportions in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Summary of missing value proportions.
    """
    validate_dataframe(df)

    return pd.DataFrame({
        "df": [
            count_missing(df).sum() / (df.shape[0] * df.shape[1])
        ],
        "var": [
            (count_missing(df) > 0).mean()
        ],
        "case": [
            (count_missing(df, axis=1) > 0).mean()
        ]
    })


def miss_scan_count(df,search=[]):
    """
    Count occurrences of specified values in each column.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    search : scalar or iterable
        Value(s) to search for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - variable
        - n
    """
    validate_dataframe(df)
    validate_search(df)

    if np.isscalar(search):
        search = [search]

    counts = {col: df[col].isin(search).sum() for col in df.columns}

    result = pd.DataFrame({
        "variable": list(counts.keys()),
        "n": list(counts.values())
    })

    return result.sort_values(by="n", ascending=False).reset_index(drop=True)

def miss_summary(df):
    """
    Generate a comprehensive summary of missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to analyze.

    Returns
    -------
    pandas.DataFrame
        A one-row summary containing:

        - miss_df_prop : float
            Proportion of missing values in the entire dataframe.

        - miss_var_prop : float
            Proportion of variables (columns) containing at least
            one missing value.

        - miss_case_prop : float
            Proportion of cases (rows) containing at least
            one missing value.

        - miss_var_summary : pandas.DataFrame
            Summary of missing values by variable.

        - miss_case_summary : pandas.DataFrame
            Summary of missing values by case.
    """
    validate_dataframe(df)

    prop_summary = miss_prop_summary(df)

    return pd.DataFrame({
        "miss_df_prop": [prop_summary.loc[0, "df"]],
        "miss_var_prop": [prop_summary.loc[0, "var"]],
        "miss_case_prop": [prop_summary.loc[0, "case"]],
        "miss_var_summary": [miss_var_summary(df)],
        "miss_case_summary": [miss_case_summary(df)]
    })