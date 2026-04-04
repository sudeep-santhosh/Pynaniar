import pandas as pd
from utils import count_missing, validate_dataframe



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

