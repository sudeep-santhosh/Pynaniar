import pandas as pd
from utils import count_missing, validate_dataframe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

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

def vis_dat(df):
    """
    Visualize missing values and column data types in a dataframe.

    This function creates a matrix-style plot where:
    - each row represents an observation
    - each column represents a dataframe column
    - each cell is colored according to the column's data type
    - missing values are highlighted in grey

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to visualize.

    Returns
    -------
    fix,ax
        so the user can further customize the plot if needed.

    Notes
    -----
    Non-missing cells are colored based on the inferred data type of
    their respective column, while missing values are always shown
    using a separate missing-value color.
    """
    validate_dataframe(df)
    n_rows, n_cols = df.shape
    plot_matrix = np.zeros((n_rows, n_cols), dtype=int)

    #dtype_list = list(df.dtypes.astype(str).values)
    dtype_list = df.dtypes.astype(str).tolist()
    unique_dtypes = list(dict.fromkeys(dtype_list))

    dtype_map = {dtype: idx for idx, dtype in enumerate(unique_dtypes)}
    for j, dtype_name in enumerate(dtype_list):
        plot_matrix[:, j] = dtype_map[dtype_name]
    na_code = len(dtype_map)

    plot_matrix[df.isna().values] = na_code
    base_colors = plt.cm.Set2.colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(unique_dtypes))]
    colors.append("grey")

    legend_handles = []
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots()
    ax.imshow(plot_matrix, cmap=cmap, aspect="auto")
    #ax.set_xticks(range(df.shape[1]))
    #ax.set_xticklabels(df.columns, rotation=45)
    plt.xticks(
    ticks=range(len(df.columns)),
    labels=df.columns,
    rotation=70
    )
    for dtype, code in dtype_map.items():
        legend_handles.append(
            Patch(facecolor=colors[code], label=dtype)
            )

    legend_handles.append(
        Patch(facecolor=colors[-1], label="NA")
        )
    
    ax.legend(handles=legend_handles, title="Type", loc="upper right")
    #plt.gca().xaxis.tick_top()
    ax.xaxis.tick_top()
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    return fig, ax


def gg_miss_upset(df):
    """
    Create an UpSet-style visualization of missing data patterns.

    This function:
    1. Identifies missing values in the dataframe
    2. Computes combinations of missingness across columns
    3. Counts frequency of each missingness pattern (intersections)
    4. Visualizes:
        - Top bar chart: size of each missingness intersection
        - Bottom matrix: pattern of missing values (dots + connecting lines)
        - Left bar chart: total missing values per column (set size)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe

    Returns
    -------
    fig, (ax_set, ax_bar, ax_matrix)
        Matplotlib figure and axes for further customization
    """

    miss = df.isna()

    patterns = miss.apply(tuple, axis=1)
    pattern_counts = patterns.value_counts()

    pattern_df = pattern_counts.reset_index()
    pattern_df.columns = ["pattern", "count"]
    pattern_matrix = pd.DataFrame(
        pattern_df["pattern"].tolist(),
        columns=df.columns
    )

    pattern_matrix["count"] = pattern_df["count"]
    pattern_matrix = pattern_matrix.astype(int)

    pattern_matrix = pattern_matrix.sort_values(
        by="count", ascending=False
    ).reset_index(drop=True)

    set_sizes = df.isna().sum()

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[2, 1])

    ax_set = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[1, 1])

    ax_bar.bar(range(len(pattern_matrix)), pattern_matrix["count"])
    ax_bar.set_ylabel("Intersection Size")
    ax_bar.set_xticks([])

    for i, row in pattern_matrix.iterrows():
        missing_positions = []

        for j, col in enumerate(df.columns):
            if row[col] == 1:
                ax_matrix.scatter(i, j, color="black", s=40)
                missing_positions.append(j)
            else:
                ax_matrix.scatter(i, j, color="lightgrey", s=40)

        if len(missing_positions) > 1:
            ax_matrix.plot(
                [i] * len(missing_positions),
                missing_positions,
                color="black"
            )

    ax_matrix.set_yticks(range(len(df.columns)))
    ax_matrix.set_yticklabels(df.columns)
    ax_matrix.set_xticks(range(len(pattern_matrix)))
    ax_matrix.set_xticklabels([])

    # --- Left bar (set size) ---
    ax_set.barh(range(len(set_sizes)), set_sizes.values)
    ax_set.set_yticks(range(len(set_sizes)))
    ax_set.set_yticklabels(set_sizes.index)
    ax_set.set_xlabel("Set Size")
    ax_set.invert_yaxis()

    # --- Clean look ---
    for ax in [ax_bar, ax_matrix, ax_set]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig, (ax_set, ax_bar, ax_matrix)


def vis_miss(df, sort=False):
    """
    Visualize missing values in a dataframe using a simple matrix plot.

    This function displays the structure of missing data where:
    - rows represent observations
    - columns represent variables
    - missing values are shown in dark color
    - present values are shown in light color

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to visualize.

    sort : bool, default=False
        If True, rows are sorted by number of missing values (descending),
        making patterns easier to detect. If False, original row order is preserved.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects for further customization.
    """

   
    miss = df.isna()


    if sort:
        miss = miss.loc[
            miss.sum(axis=1).sort_values(ascending=False).index
        ]

    plot_matrix = miss.astype(int)


    fig, ax = plt.subplots(figsize=(10, 6))

    ax.imshow(plot_matrix, aspect="auto", cmap="gray_r")

  
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=70)

    ax.set_yticks([]) 
    ax.set_title("Missing Data Matrix")

    plt.tight_layout()

    return fig, ax




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

