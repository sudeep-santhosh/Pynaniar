import pandas as pd
from utils import count_missing
from validation import validate_dataframe, validate_column, validation_span_check
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

def gg_miss_span(df, var, span_every, facet=None, visualizer="mat"):
    """
    Visualize the proportion of missing values in repeated spans of a variable.

    This mirrors the behaviour of naniar's gg_miss_span by summarizing a
    variable into contiguous windows of length ``span_every`` and showing the
    proportion of missing vs complete values in each span as a stacked 100% bar chart.
    The default visualization uses matplotlib, but seaborn and plotly are also
    supported through the ``visualizer`` argument.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    var : str
        Name of the variable to examine for missing values.
    span_every : int
        Size of each span/window.
    facet : str, optional
        Optional column name used to create faceted subplots.
    visualizer : str, default="mat"
        Plotting backend. Supported values are ``"mat"``, ``"sb"`` and
        ``"plotly"``.

    Returns
    -------
    fig, ax
        Figure object and axes for matplotlib/seaborn, or a plotly figure and
        ``None`` for plotly.
    """
    validate_dataframe(df)
    validate_column(df, var)
    validation_span_check(span_every)

    if facet is not None:
        validate_column(df, facet)

    series = df[var]
    if not isinstance(series, pd.Series):
        raise TypeError("'var' must refer to a pandas Series")

    visualizer = visualizer.lower()

    def build_span_df(values):
        n_rows = len(values)
        span_counter = np.arange(1, n_rows // span_every + 2)
        spans = []
        for start in range(0, n_rows, span_every):
            end = min(start + span_every, n_rows)
            spans.append(values.iloc[start:end])

        n_miss_list = []
        n_complete_list = []
        for span in spans:
            n_miss = span.isna().sum()
            n_complete = (~span.isna()).sum()
            n_miss_list.append(n_miss)
            n_complete_list.append(n_complete)

        total = [n_m + n_c for n_m, n_c in zip(n_miss_list, n_complete_list)]
        prop_miss = [n_m / t if t > 0 else 0 for n_m, t in zip(n_miss_list, total)]
        prop_complete = [n_c / t if t > 0 else 0 for n_c, t in zip(n_complete_list, total)]

        return pd.DataFrame({
            "span_counter": span_counter[:len(n_miss_list)],
            "n_miss": n_miss_list,
            "n_complete": n_complete_list,
            "prop_miss": prop_miss,
            "prop_complete": prop_complete,
        })

    if facet is None:
        span_df = build_span_df(series)

        if visualizer == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=span_df["span_counter"],
                y=span_df["prop_complete"],
                name="Present",
                marker_color="lightgrey",
            ))
            fig.add_trace(go.Bar(
                x=span_df["span_counter"],
                y=span_df["prop_miss"],
                name="Missing",
                marker_color="black",
            ))
            fig.update_layout(
                barmode="stack",
                xaxis_title="Span",
                yaxis_title="Proportion missing",
                title=f"Proportion of missing values<br>Over a repeating span of {span_every}",
                yaxis=dict(tickformat=".0%"),
            )
            return fig, None

        fig, ax = plt.subplots(figsize=(8, 4))
        x_pos = np.arange(len(span_df))
        
        if visualizer == "sb":
            import seaborn as sns
        
        ax.bar(x_pos, span_df["prop_complete"], label="Present", color="lightgrey")
        ax.bar(x_pos, span_df["prop_miss"], bottom=span_df["prop_complete"], label="Missing", color="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(span_df["span_counter"])

        ax.set_xlabel("Span")
        ax.set_ylabel("Proportion missing")
        ax.set_title(f"Proportion of missing values\nOver a repeating span of {span_every}")
        ax.legend(loc="upper right")
        ax.set_ylim([0, 1])
        plt.tight_layout()
        return fig, ax

    facet_values = df[facet].astype(str)
    facet_levels = sorted(facet_values.unique())

    if visualizer == "plotly":
        import plotly.graph_objects as go
        import plotly.subplots as sp

        records = []
        for facet_value in facet_levels:
            subset = df[facet_values == facet_value]
            subset_span_df = build_span_df(subset[var])
            subset_span_df[facet] = facet_value
            records.append(subset_span_df)

        plot_df = pd.concat(records, ignore_index=True)
        
        fig = sp.make_subplots(rows=1, cols=len(facet_levels), subplot_titles=facet_levels)
        for i, facet_value in enumerate(facet_levels, 1):
            subset_df = plot_df[plot_df[facet] == facet_value]
            fig.add_trace(
                go.Bar(x=subset_df["span_counter"], y=subset_df["prop_complete"], name="Present", marker_color="lightgrey", showlegend=(i==1)),
                row=1, col=i,
            )
            fig.add_trace(
                go.Bar(x=subset_df["span_counter"], y=subset_df["prop_miss"], name="Missing", marker_color="black", showlegend=(i==1)),
                row=1, col=i,
            )
        fig.update_layout(barmode="stack", title_text=f"Proportion of missing values by {facet}<br>Over a repeating span of {span_every}", height=400)
        fig.update_yaxes(tickformat=".0%")
        return fig, None

    fig, axes = plt.subplots(
        nrows=len(facet_levels),
        ncols=1,
        figsize=(8, 3 * len(facet_levels)),
        sharex=True,
    )
    if len(facet_levels) == 1:
        axes = [axes]

    for ax, facet_value in zip(axes, facet_levels):
        subset = df[facet_values == facet_value]
        subset_span_df = build_span_df(subset[var])

        x_pos = np.arange(len(subset_span_df))
        ax.bar(x_pos, subset_span_df["prop_complete"], label="Present", color="lightgrey")
        ax.bar(x_pos, subset_span_df["prop_miss"], bottom=subset_span_df["prop_complete"], label="Missing", color="black")
        
        ax.set_title(f"{facet}={facet_value}")
        ax.set_ylabel("Proportion missing")
        ax.set_xlabel("Span")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subset_span_df["span_counter"])
        ax.set_ylim([0, 1])
        if ax == axes[0]:
            ax.legend(loc="upper right")

    fig.suptitle(f"Proportion of missing values by {var}\nOver a repeating span of {span_every}")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, axes


def geom_miss_point(df, x, y, visualizer="mat", prop_below=0.1, jitter=0.05):
    """
    Plot a scatter plot that highlights missing values in one variable.

    This mirrors the intent of naniar's ``geom_miss_point()`` by showing a
    standard point plot for two variables while visually distinguishing missing
    values in the y variable. Missing points are drawn with a different color
    and a small vertical offset to make them visible.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Name of the x-axis variable.
    y : str
        Name of the y-axis variable.
    visualizer : str, default="mat"
        Plotting backend. Supported values are ``"mat"``, ``"sb"`` and
        ``"plotly"``.
    prop_below : float, default=0.1
        Vertical offset applied to missing points.
    jitter : float, default=0.05
        Small random jitter added to missing-point positions to avoid overlap.

    Returns
    -------
    fig, ax
        Figure and axes for matplotlib/seaborn, or a plotly figure and None.
    """
    validate_dataframe(df)
    validate_column(df, x)
    validate_column(df, y)

    visualizer = visualizer.lower()

    def impute_below(series, prop_below, jitter, seed_shift=159):
        values = series.astype(float).copy()
        mask = values.isna()
        if not mask.any():
            return values

        complete = values[~mask]
        if complete.empty:
            return values

        if len(complete) == 1 or complete.var() == 0:
            xmin = complete.min()
            x_shift = xmin - xmin * prop_below
            jitter_mag = abs(xmin) * jitter if xmin != 0 else jitter
        else:
            xmin = complete.min()
            xrange = complete.max() - complete.min()
            x_shift = xmin - xrange * prop_below
            jitter_mag = xrange * jitter

        rng = np.random.RandomState(seed_shift)
        jitter_values = (rng.rand(len(values)) - 0.5) * jitter_mag

        values.loc[mask] = x_shift + jitter_values[mask]
        return values

    plot_df = df[[x, y]].copy().reset_index(drop=True)
    plot_df["missing"] = np.where(plot_df[x].isna() | plot_df[y].isna(), "Missing", "Not Missing")
    plot_df["x_plot"] = impute_below(plot_df[x], prop_below, jitter)
    plot_df["y_plot"] = impute_below(plot_df[y], prop_below, jitter)

    palette = {"Not Missing": "#00BFC4", "Missing": "#F8766D"}

    if visualizer == "plotly":
        import plotly.express as px

        fig = px.scatter(
            plot_df,
            x="x_plot",
            y="y_plot",
            color="missing",
            color_discrete_map=palette,
            labels={"x_plot": x, "y_plot": y, "missing": "missing"},
            title=f"Missing values for {y} against {x}",
        )
        fig.update_traces(marker=dict(size=8))
        return fig, None

    fig, ax = plt.subplots(figsize=(8, 4))

    if visualizer == "sb":
        import seaborn as sns

        sns.scatterplot(
            data=plot_df,
            x="x_plot",
            y="y_plot",
            hue="missing",
            palette=palette,
            ax=ax,
            s=50,
            legend="full",
        )
    elif visualizer == "mat":
        for label, color in palette.items():
            subset = plot_df[plot_df["missing"] == label]
            ax.scatter(
                subset["x_plot"],
                subset["y_plot"],
                color=color,
                label=label,
                s=50,
                alpha=0.8,
                edgecolors="face",
                linewidths=0,
            )
    else:
        print(f"Unknown visualizer '{visualizer}', defaulting to 'mat'")
        for label, color in palette.items():
            subset = plot_df[plot_df["missing"] == label]
            ax.scatter(
                subset["x_plot"],
                subset["y_plot"],
                color=color,
                label=label,
                s=50,
                alpha=0.8,
                edgecolors="face",
                linewidths=0,
            )

    ax.set_title(f"Missing values for {y} against {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def vis_dat(df, visualizer="mat"):
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
    
    if visualizer == "plotly":
        # convert matplotlib colors to hex for plotly
        hex_colors = [to_hex(c) for c in colors]
        import plotly.express as px
        fig = px.imshow(plot_matrix, color_continuous_scale=hex_colors, aspect='auto')
        fig.update_coloraxes(showscale=False)
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(df.columns))),
            ticktext=list(df.columns),
            tickangle=70,
            side='top'
        )
        ax = None
    else:
        fig, ax = plt.subplots()
        if visualizer == "mat":
            ax.imshow(plot_matrix, cmap=cmap, aspect="auto")
        elif visualizer == "sb":
            import seaborn as sns
            sns.heatmap(plot_matrix, cmap=cmap, ax=ax, cbar=False)
        else:
            print(f"Unknown visualizer '{visualizer}', defaulting to 'mat'")
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


def gg_miss_upset(df, visualizer="mat"):
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
    validate_dataframe(df)
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

    if visualizer == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.35, 0.65],
            row_heights=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        )
        # Top bar: intersection sizes (row 1, col 2)
        fig.add_trace(
            go.Bar(x=list(range(len(pattern_matrix))), y=pattern_matrix["count"].tolist(), name="Intersection Size", marker_color="steelblue"),
            row=1,
            col=2,
        )

        # Left bar: set sizes (row 2, col 1)
        fig.add_trace(
            go.Bar(x=set_sizes.values.tolist(), y=set_sizes.index.tolist(), orientation="h", name="Set Size", marker_color="steelblue"),
            row=2,
            col=1,
        )

        # Matrix: dots + connecting lines (row 2, col 2)
        matrix_df = pattern_matrix.drop(columns=["count"])  # columns = original df columns
        for i, row in matrix_df.iterrows():
            xs = []
            ys = []
            colors = []
            for j, col in enumerate(matrix_df.columns):
                xs.append(i)
                ys.append(j)
                colors.append("black" if row[col] == 1 else "lightgrey")
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=colors, size=8), showlegend=False),
                row=2,
                col=2,
            )
            missing_positions = [j for j, col in enumerate(matrix_df.columns) if row[col] == 1]
            if len(missing_positions) > 1:
                fig.add_trace(
                    go.Scatter(x=[i] * len(missing_positions), y=missing_positions, mode="lines", line=dict(color="black"), showlegend=False),
                    row=2,
                    col=2,
                )

        # Axes configuration
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(df.columns))), ticktext=list(df.columns), row=2, col=2)
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(pattern_matrix))), ticktext=[], row=2, col=2)
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(pattern_matrix))), ticktext=list(range(len(pattern_matrix))), row=1, col=2)

        fig.update_layout(title="Missing Data Patterns", showlegend=False)
        return fig, None
    else:
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[2, 1])

        ax_set = fig.add_subplot(gs[1, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_matrix = fig.add_subplot(gs[1, 1])

        if visualizer == "sb":
            import seaborn as sns
            sns.barplot(x=list(range(len(pattern_matrix))), y=pattern_matrix["count"], ax=ax_bar, color="steelblue")
        # Default to Matplotlib when visualizer is 'mat' or unknown
        elif visualizer == "mat":
            ax_bar.bar(range(len(pattern_matrix)), pattern_matrix["count"])
        else:
            ax_bar.bar(range(len(pattern_matrix)), pattern_matrix["count"])
            print(f"Unknown visualizer '{visualizer}', defaulting to 'mat'")

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


def vis_miss(df, sort=False, visualizer="mat"):
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

    if visualizer == "plotly":
        import plotly.express as px
        fig = px.imshow(plot_matrix, color_continuous_scale=["lightgray", "black"], aspect="auto")
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(df.columns))),
            ticktext=list(df.columns),
            tickangle=70,
            side="top",
        )
        fig.update_yaxes(showticklabels=False)
        return fig, None
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        if visualizer == "mat":
            ax.imshow(plot_matrix, aspect="auto", cmap="gray_r")
        elif visualizer == "sb":
            import seaborn as sns
            sns.heatmap(plot_matrix, ax=ax, cmap="gray_r", cbar=False)
        else:
            print(f"Unknown visualizer '{visualizer}', defaulting to 'mat'")
            ax.imshow(plot_matrix, aspect="auto", cmap="gray_r")

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=70)

        ax.set_yticks([])
        ax.set_title("Missing Data Matrix")

        plt.tight_layout()

        return fig, ax
