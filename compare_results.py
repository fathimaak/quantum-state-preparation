
"""
compare_results.py - Integration and results visualization (owner: Donita)

Reads the detailed CSVs from Sankhana (destructive SWAP) and Haneen
(classical shadows), checks that they are comparable, and produces:

    results/comparison_table.csv
    results/comparison_table.md
    results/comparison_plots.png
    results/headline_table.png

Run:
    python compare_results.py

Requirements:
    pandas, numpy, matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HERE = Path(__file__).parent
RES = HERE / "results"

SWAP_CSV = RES / "destructive_swap_results.csv"
SHAD_CSV = RES / "shadows_results.csv"

REFERENCE = 0.25

NAMES = {
    "destructive_swap": "Destructive SWAP",
    "classical_shadows": "Classical Shadows",
}

COLORS = {
    "destructive_swap": "#1f77b4",
    "classical_shadows": "#d95f02",
}


# ---------------------------------------------------------
# Load and validate experiment results
# ---------------------------------------------------------

def load_and_check():
    """Load both experiment CSV files and verify comparability."""

    if not SWAP_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {SWAP_CSV}")

    if not SHAD_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {SHAD_CSV}")

    swap_df = pd.read_csv(SWAP_CSV)
    shadows_df = pd.read_csv(SHAD_CSV)

    df = pd.concat(
        [swap_df, shadows_df],
        ignore_index=True
    )

    required_columns = {
        "method",
        "shots",
        "estimate",
        "theoretical",
        "abs_error",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in experiment CSVs: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The experiment CSV files contain no results.")

    # 1. Both methods must use the same theoretical reference.
    assert np.allclose(
        df["theoretical"],
        REFERENCE,
        atol=1e-9
    ), "The theoretical overlap differs from 0.25."

    # 2. Both methods must use the same shot budgets and trial counts.
    counts = df.groupby(
        ["method", "shots"]
    ).size().unstack("method")

    assert counts.notna().all().all(), (
        "The methods do not have matching shot budgets."
    )

    assert (counts.nunique(axis=1) == 1).all(), (
        "The methods have different numbers of trials at one or more budgets."
    )

    # 3. Verify the recorded absolute errors.
    expected_abs_error = (df["estimate"] - REFERENCE).abs()

    assert np.allclose(
        df["abs_error"],
        expected_abs_error,
        atol=1e-9
    ), "The abs_error column is inconsistent with the estimates."

    # 4. Confirm that both expected methods are present.
    expected_methods = set(NAMES.keys())
    actual_methods = set(df["method"].unique())

    assert actual_methods == expected_methods, (
        f"Expected methods {expected_methods}, but found {actual_methods}."
    )

    print("Validation passed.")
    print(f"Theoretical overlap: {REFERENCE:.4f}")
    print(f"Shot budgets: {sorted(df['shots'].unique())}")

    return df


# ---------------------------------------------------------
# Summarize results
# ---------------------------------------------------------

def summarize(df):
    """Calculate summary statistics for each method and shot budget."""

    rows = []

    for (method, shots), group in df.groupby(["method", "shots"]):

        estimates = group["estimate"].to_numpy()
        absolute_errors = np.abs(estimates - REFERENCE)

        rows.append({
            "method": method,
            "shots": shots,
            "trials": len(group),

            "mean_estimate": estimates.mean(),

            "std": estimates.std(ddof=1),

            "mean_abs_error": absolute_errors.mean(),

            # Standard error of the mean absolute error.
            "mae_sem": (
                absolute_errors.std(ddof=1) / np.sqrt(len(absolute_errors))
                if len(absolute_errors) > 1
                else 0.0
            ),

            # Root mean squared error.
            "rmse": np.sqrt(
                np.mean((estimates - REFERENCE) ** 2)
            ),

            # Absolute error of the mean estimate.
            "abs_error_of_mean": abs(
                estimates.mean() - REFERENCE
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Build side-by-side comparison table
# ---------------------------------------------------------

def wide_table(summary):
    """Create one comparison row for each shot budget."""

    swap = (
        summary[summary["method"] == "destructive_swap"]
        .set_index("shots")
        .sort_index()
    )

    shadows = (
        summary[summary["method"] == "classical_shadows"]
        .set_index("shots")
        .sort_index()
    )

    # Ensure both methods have matching shot budgets.
    assert swap.index.equals(shadows.index), (
        "SWAP and Classical Shadows shot budgets do not match."
    )

    table = pd.DataFrame({
        "Shots": swap.index,

        "Theoretical": REFERENCE,

        "SWAP mean est.": swap["mean_estimate"].values,

        "Shadows mean est.": shadows["mean_estimate"].values,

        "SWAP mean |err|": swap["mean_abs_error"].values,

        "Shadows mean |err|": shadows["mean_abs_error"].values,

        "SWAP RMSE": swap["rmse"].values,

        "Shadows RMSE": shadows["rmse"].values,
    })

    swap_error = table["SWAP mean |err|"]
    shadows_error = table["Shadows mean |err|"]

    # Ratio above 1 means Shadows had greater observed mean absolute error.
    ratio = shadows_error / swap_error.replace(0, np.nan)

    # Use "~tie" only when observed errors are within 10% of each other.
    better = np.where(
        swap_error < shadows_error,
        "SWAP",
        "Shadows"
    )

    table["Lower error"] = np.where(
        (ratio >= 0.9) & (ratio <= 1.1),
        "~tie",
        better
    )

    table["Shadows/SWAP error"] = ratio

    return table


# ---------------------------------------------------------
# Generate comparison plots
# ---------------------------------------------------------

def make_plots(df, summary, table):
    """Create the three-panel comparison figure."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4.8)
    )

    shots = np.array(sorted(df["shots"].unique()))

    trial_count = int(summary["trials"].iloc[0])

    # Panel (a): Mean absolute error versus shots.
    for method in NAMES:

        group = (
            summary[summary["method"] == method]
            .sort_values("shots")
        )

        axes[0].errorbar(
            group["shots"],
            group["mean_abs_error"],
            yerr=group["mae_sem"],
            marker="o",
            capsize=3,
            color=COLORS[method],
            label=NAMES[method]
        )

    # Reference guide illustrating approximate 1/sqrt(N) scaling.
    # This is a visual guide, not a fitted model.
    reference_line = (
        table["SWAP mean |err|"].iloc[0]
        * np.sqrt(shots[0] / shots)
    )

    axes[0].plot(
        shots,
        reference_line,
        "k--",
        linewidth=1,
        alpha=0.5,
        label=r"$1/\sqrt{N}$ guide"
    )

    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Total shots",
        ylabel="Mean absolute error",
        title="(a) Absolute error vs shots"
    )

    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    # Panel (b): Mean estimate with plus/minus one standard deviation.
    for method, offset in zip(NAMES, (0.97, 1.03)):

        group = (
            summary[summary["method"] == method]
            .sort_values("shots")
        )

        axes[1].errorbar(
            group["shots"] * offset,
            group["mean_estimate"],
            yerr=group["std"],
            marker="o",
            capsize=3,
            color=COLORS[method],
            label=NAMES[method]
        )

    axes[1].axhline(
        REFERENCE,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Theoretical = {REFERENCE:.2f}"
    )

    axes[1].set(
        xscale="log",
        xlabel="Total shots",
        ylabel="Estimated overlap",
        title=f"(b) Mean estimate ± 1 std ({trial_count} trials)"
    )

    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel (c): Individual estimates at 1,000 and 10,000 shots.
    positions = []
    data = []
    box_colors = []
    labels = []

    selected_budgets = [1000, 10000]

    for i, shot_budget in enumerate(selected_budgets):

        for j, method in enumerate(NAMES):

            values = df.loc[
                (df["method"] == method)
                & (df["shots"] == shot_budget),
                "estimate"
            ].to_numpy()

            if len(values) == 0:
                raise ValueError(
                    f"No results found for {method} at {shot_budget} shots."
                )

            positions.append(i * 3 + j)
            data.append(values)
            box_colors.append(COLORS[method])

        labels.append(f"{shot_budget} shots")

    boxplot = axes[2].boxplot(
        data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False
    )

    rng = np.random.default_rng(0)

    for patch, color, position, values in zip(
        boxplot["boxes"],
        box_colors,
        positions,
        data
    ):

        patch.set(
            facecolor=color,
            alpha=0.35
        )

        # Small horizontal jitter makes overlapping trial points visible.
        jitter = rng.uniform(
            -0.18,
            0.18,
            len(values)
        )

        axes[2].scatter(
            position + jitter,
            values,
            s=12,
            color=color,
            zorder=3
        )

    axes[2].axhline(
        REFERENCE,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    axes[2].set_xticks([0.5, 3.5])
    axes[2].set_xticklabels(labels)

    axes[2].set(
        ylabel="Estimated overlap",
        title="(c) Spread of individual trials"
    )

    axes[2].grid(
        alpha=0.3,
        axis="y"
    )

    axes[2].legend(
        handles=[
            Patch(
                color=COLORS[method],
                alpha=0.5,
                label=NAMES[method]
            )
            for method in NAMES
        ]
    )

    fig.suptitle(
        "SWAP or Shadows? Overlap of State A and State B "
        f"(theoretical {REFERENCE:.2f}, ideal Aer simulation)",
        fontsize=12
    )

    fig.tight_layout()

    fig.savefig(
        RES / "comparison_plots.png",
        dpi=170,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------
# Generate slide-ready headline table
# ---------------------------------------------------------

def make_table_image(table):
    """Generate a readable table image for presentations."""

    columns = [
        "Shots",
        "SWAP mean est.",
        "Shadows mean est.",
        "SWAP mean |err|",
        "Shadows mean |err|",
        "Lower error",
    ]

    cells = [
        [
            f"{int(row['Shots'])}",
            f"{row['SWAP mean est.']:.4f}",
            f"{row['Shadows mean est.']:.4f}",
            f"{row['SWAP mean |err|']:.4f}",
            f"{row['Shadows mean |err|']:.4f}",
            row["Lower error"],
        ]
        for _, row in table.iterrows()
    ]

    # Read trial count directly from the experiment CSV.
    # Each method has one row per trial at each shot budget.
    swap_data = pd.read_csv(SWAP_CSV)
    trial_count = int(
        swap_data.groupby("shots").size().iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(13, 0.65 * len(cells) + 1.5)
    )

    ax.axis("off")

    table_artist = ax.table(
        cellText=cells,
        colLabels=columns,
        loc="center",
        cellLoc="center"
    )

    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1.15, 1.6)

    # Style the header row.
    for column_index in range(len(columns)):
        table_artist[0, column_index].set_facecolor("#333333")
        table_artist[0, column_index].set_text_props(
            color="white",
            weight="bold"
        )

    ax.set_title(
        f"Theoretical overlap = {REFERENCE:.2f}  |  "
        f"means over {trial_count} trials per budget",
        fontsize=11,
        pad=15
    )

    fig.savefig(
        RES / "headline_table.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------
# Write Markdown comparison table
# ---------------------------------------------------------

def write_markdown(table):

    lines = [
        "| Shots | Theoretical | SWAP mean est. | Shadows mean est. | "
        "SWAP mean abs err | Shadows mean abs err | Lower error |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for _, row in table.iterrows():

        lines.append(
            f"| {int(row['Shots'])} "
            f"| {row['Theoretical']:.2f} "
            f"| {row['SWAP mean est.']:.4f} "
            f"| {row['Shadows mean est.']:.4f} "
            f"| {row['SWAP mean |err|']:.4f} "
            f"| {row['Shadows mean |err|']:.4f} "
            f"| {row['Lower error']} |"
        )

    output_path = RES / "comparison_table.md"

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # Ensure the output directory exists.
    RES.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load, validate, and summarize the experiment results.
    df = load_and_check()

    summary = summarize(df)

    table = wide_table(summary)

    # Save comparison outputs.
    table.to_csv(
        RES / "comparison_table.csv",
        index=False
    )

    write_markdown(table)

    make_plots(
        df,
        summary,
        table
    )

    make_table_image(table)

    # Print the comparison table.
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print("\nComparison summary:")
    print(table.round(4).to_string(index=False))

    print("\nAll comparison outputs were written to:")
    print(RES.resolve())


if __name__ == "__main__":
    main()

