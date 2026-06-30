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

def miss_var_cumsum(df):
    """
    Calculate cumulative sum of missing values across variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing:

        - variable : column name
        - n_miss : number of missing values in the variable
        - n_miss_cumsum : cumulative sum of missing values
    """
    validate_dataframe(df)

    result = pd.DataFrame({
        "variable": df.columns,
        "n_miss": count_missing(df).values
    })

    result["n_miss_cumsum"] = result["n_miss"].cumsum()

    return result

def miss_var_run(df, var):
    """
    Compute run lengths of missing and complete values
    for a single variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    var : str
        Column name.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - run_length
        - is_na
    """
    validate_dataframe(df)
    validate_search(var)

    is_missing = df[var].isna()
    runs = []
    current = is_missing.iloc[0]
    length = 1

    for value in is_missing.iloc[1:]:
        if value == current:
            length += 1
        else:
            runs.append({
                "run_length": length,
                "is_na": "missing" if current else "complete"
            })
            current = value
            length = 1

    runs.append({
        "run_length": length,
        "is_na": "missing" if current else "complete"
    })

    return pd.DataFrame(runs)

def miss_var_span(df, var, span_every):
    """
    Summarize missingness in consecutive spans of a variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    var : str
        Column name.

    span_every : int
        Number of observations per span.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - span_counter
        - n_miss
        - n_complete
        - prop_miss
        - prop_complete
        - n_in_span
    """
    validate_dataframe(df)
    validate_column(df, var)
    validation_span_check(span_every)
    series = df[var]
    results = []
    for i in range(0, len(df), span_every):

        chunk = df[[var]].iloc[i:i + span_every]

        summary = miss_var_summary(chunk)

        n_miss = summary.loc[0, "n_miss"]
        n_complete = summary.loc[0, "n_complete"]
        pct_miss = summary.loc[0, "pct_miss"]

        results.append({
            "span_counter": len(results) + 1,
            "n_miss": n_miss,
            "n_complete": n_complete,
            "prop_miss": pct_miss / 100,
            "prop_complete": 1 - (pct_miss / 100),
            "n_in_span": len(chunk)
        })

    return pd.DataFrame(results)

def miss_var_which(df):
    """
    Return the names of variables containing at least one missing value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    list
        List of column names containing missing values.
    """
    validate_dataframe(df)

    return df.columns[df.isna().any()].tolist()

def n_complete(x):
    """
    Count the number of complete (non-missing) values.

    Parameters
    ----------
    x : pandas.Series, numpy.ndarray, list
        Input vector.

    Returns
    -------
    int
        Number of complete values.
    """

    validation_x(x)
    return pd.Series(x).notna().sum()

def n_miss(x):
    """
    Count the number of missing values.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    int
        Number of missing values.
    """

    validation_x(x)
    return len(x) - n_complete(x)

def n_miss_row(df):
    """
    Count the number of missing values in each row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.Series
        Number of missing values in each row.
    """
    validate_dataframe(df)

    return df.isna().sum(axis=1)

def n_complete_row(df):
    """
    Count the number of complete (non-missing) values in each row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.Series
        Number of complete values in each row.
    """
    validate_dataframe(df)

    return df.shape[1] - n_miss_row(df)

def pct_complete(x):
    """
    Calculate the percentage of complete (non-missing) values.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    float
        Percentage of complete values in the input vector.
    """
    validation_x(x)

    return (n_complete(x) / len(x)) * 100

def pct_miss(x):
    """
    Calculate the percentage of missing values.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    float
        Percentage of missing values in the input vector.
    """
    validation_x(x)

    return 100 - pct_complete(x)

def prop_complete(x):
    """
    Calculate the proportion of complete (non-missing) values.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    float
        Proportion of complete values in the input vector.
    """
    validation_x(x)

    return n_complete(x) / len(x)

def prop_miss(x):
    """
    Calculate the proportion of missing values.

    Parameters
    ----------
    x : array-like
        Input vector.

    Returns
    -------
    float
        Proportion of missing values in the input vector.
    """
    validation_x(x)

    return 1 - prop_complete(x)

def prop_complete_row(df):
    """
    Calculate the proportion of complete values in each row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.Series
        Proportion of complete values in each row.
    """
    validate_dataframe(df)

    return n_complete_row(df) / df.shape[1]

def prop_miss_row(df):
    """
    Calculate the proportion of missing values in each row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.Series
        Proportion of missing values in each row.
    """
    validate_dataframe(df)

    return 1 - prop_complete_row(df)


def add_any_miss(
    df,
    columns=None,
    label="any_miss",
    missing="missing",
    complete="complete"
):
    """
    Add a column indicating whether each row contains
    at least one missing value.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list-like, optional
        Columns to check for missing values.
        If None, all columns are used.
    label : str, default="any_miss"
        Name of the new column.
    missing : str, default="missing"
        Value assigned when a row contains at least
        one missing value.
    complete : str, default="complete"
        Value assigned when a row contains no
        missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional indicator column.
    """
    validate_dataframe(df)

    result = df.copy()

    if columns is None:
        check_df = result
        suffix = "_all"
    else:
        validate_columns(df, columns)

        check_df = result[columns]
        suffix = "_vars"

    result[f"{label}{suffix}"] = np.where(
        check_df.isna().any(axis=1),
        missing,
        complete
    )

    return result

def add_n_miss(df, columns=None, label="n_miss"):
    """
    Add a column containing the number of missing values per row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    columns : list[str], optional
        Columns to use when counting missing values.
        If None, all columns are used.

    label : str, default="n_miss"
        Base name for the added column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional column containing
        row-wise missing value counts.
    """
    validate_dataframe(df)

    result = df.copy()

    if columns is None:
        result[f"{label}_all"] = n_miss_row(df)
    else:
        validate_columns(df, columns)
        result[f"{label}_vars"] = n_miss_row(df[columns])

    return result

def add_prop_miss(df, columns=None, label="prop_miss"):
    """
    Add a column containing the proportion of missing values per row.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    columns : list[str], optional
        Columns to use when calculating the proportion
        of missing values. If None, all columns are used.

    label : str, default="prop_miss"
        Base name for the added column.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an additional column containing
        row-wise proportions of missing values.
    """
    validate_dataframe(df)

    result = df.copy()

    if columns is None:
        result[f"{label}_all"] = prop_miss_row(df)
    else:
        validate_columns(df, columns)
        result[f"{label}_vars"] = prop_miss_row(df[columns])

    return result

def any_row_miss(df):
    """
    Check whether each row contains at least one missing value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.Series
        Boolean Series indicating whether each row
        contains at least one missing value.
    """
    validate_dataframe(df)

    return df.isna().any(axis=1)

def label_missings(df, columns=None,
                   missing="Missing",
                   complete="Not Missing"):
    """
    Label rows as missing or complete.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    columns : list[str], optional
        Columns to inspect. If None, all columns are used.

    missing : str, default="Missing"
        Label for rows containing at least one missing value.

    complete : str, default="Not Missing"
        Label for rows containing no missing values.

    Returns
    -------
    pandas.Series
        Series containing missing/complete labels.
    """
    validate_dataframe(df)

    if columns is None:
        data = df
    else:
        validate_columns(df, columns)
        data = df[columns]

    return np.where(
        any_row_miss(data),
        missing,
        complete
    )


def add_label_missings(df,
                       columns=None,
                       missing="Missing",
                       complete="Not Missing"):
    """
    Add a column indicating whether a row contains
    any missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    columns : list[str], optional
        Columns to inspect. If None, all columns are used.

    missing : str, default="Missing"
        Label for rows containing at least one missing value.

    complete : str, default="Not Missing"
        Label for rows containing no missing values.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an added 'any_missing' column.
    """
    validate_dataframe(df)

    result = df.copy()

    result["any_missing"] = label_missings(
        result,
        columns=columns,
        missing=missing,
        complete=complete
    )

    return result

def any_row_shade(df):
    """
    Check whether each row contains at least one shadow
    missing value.

    Parameters
    ----------
    df : pandas.DataFrame
        Shadow dataframe.

    Returns
    -------
    pandas.Series
        Boolean Series indicating whether each row
        contains at least one shadow missing value.
    """
    validate_dataframe(df)

    return df.astype(str).apply(
        lambda row: row.str.match(r"^NA|^NA_").any(),
        axis=1
    )

def label_shadow(
    df,
    columns=None,
    missing="Missing",
    complete="Not Missing"
):
    """
    Label rows in shadow data as missing or complete.

    Parameters
    ----------
    df : pandas.DataFrame
        Shadow dataframe.

    columns : list[str], optional
        Columns to inspect. If None, all columns
        are used.

    missing : str, default="Missing"
        Label assigned to rows containing at least
        one shadow missing value.

    complete : str, default="Not Missing"
        Label assigned to rows containing no shadow
        missing values.

    Returns
    -------
    pandas.Series
        Series of missing/complete labels.
    """
    validate_dataframe(df)

    if not any_shadow(df):
        raise ValueError(
            "label_shadow() requires shadow columns."
        )

    if columns is not None:
        validate_columns(df, columns)
        df = df[columns]

    return np.where(
        any_row_shade(df),
        missing,
        complete
    )

def add_label_shadow(
    df,
    columns=None,
    missing="Missing",
    complete="Not Missing"
):
    """
    Add a column indicating whether each row
    contains shadow missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Shadow dataframe.

    columns : list[str], optional
        Columns to inspect. If None, all columns
        are used.

    missing : str, default="Missing"
        Label assigned to rows containing at least
        one shadow missing value.

    complete : str, default="Not Missing"
        Label assigned to rows containing no shadow
        missing values.

    Returns
    -------
    pandas.DataFrame
        Dataframe with an added 'any_missing'
        column.
    """
    validate_dataframe(df)

    if not any_shadow(df):
        raise ValueError(
            "add_label_shadow() requires shadow columns."
        )

    result = df.copy()

    result["any_missing"] = label_shadow(
        result,
        columns=columns,
        missing=missing,
        complete=complete
    )

    return result
