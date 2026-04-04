import pandas as pd

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

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    

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


    