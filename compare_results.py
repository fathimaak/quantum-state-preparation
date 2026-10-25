"""
compare_results.py - Integration and results visualization (owner: Donita)

Reads the detailed CSVs from Sankhana (destructive SWAP) and Haneen
(classical shadows), checks that they are comparable, and produces:

    results/comparison_table.csv   side-by-side table, one row per shot budget
    results/comparison_table.md    the same table, ready to paste into the report
    results/comparison_plots.png   three-panel figure
    results/headline_table.png     compact table image for slides

Run:  python compare_results.py          (needs pandas, numpy, matplotlib)

Metrics (theta = 0.25 is the theoretical overlap):
    mean |err|         = mean(|estimate - theta|) over the trials
    RMSE               = sqrt(mean((estimate - theta)^2))
    abs_error_of_mean  = |mean(estimate) - theta|   (not the same as mean |err|)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------- configuration
HERE = Path(__file__).parent
RES = HERE / "results"
SWAP_CSV = RES / "destructive_swap_results.csv"
SHAD_CSV = RES / "shadows_results.csv"

REFERENCE = 0.25        # theoretical overlap from states.py (Fathima)
TOLERANCE = 1e-9        # numerical tolerance for the reference and abs_error checks
EXPECTED_SHOTS = [100, 200, 500, 1000, 2000, 5000, 10000]
EXPECTED_TRIALS = 100   # trials per method and shot budget; change deliberately if the experiment changes
TIE_BAND = 0.10         # display heuristic only, see wide_table()

NAMES = {"destructive_swap": "Destructive SWAP", "classical_shadows": "Classical Shadows"}
COLORS = {"destructive_swap": "#1f77b4", "classical_shadows": "#d95f02"}

REQUIRED_COLUMNS = ["method", "n", "shots", "trial", "estimate", "theoretical", "abs_error"]
NUMERIC_COLUMNS = [c for c in REQUIRED_COLUMNS if c != "method"]


# ---------------------------------------------------------------- load and validate
def read_csv_checked(path):
    """Read one results CSV and check that it is non-empty, complete and numeric."""
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} contains no results.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(df[NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name} has non-numeric, NaN or infinite values.")
    df[["n", "shots", "trial"]] = df[["n", "shots", "trial"]].astype(int)
    return df


def load_and_check():
    """Load both CSVs and verify that the two methods are directly comparable."""
    df = pd.concat([read_csv_checked(SWAP_CSV), read_csv_checked(SHAD_CSV)], ignore_index=True)

    if set(df["method"]) != set(NAMES):
        raise ValueError(f"Expected methods {sorted(NAMES)}, found {sorted(set(df['method']))}.")
    if df["n"].nunique() != 1:
        raise ValueError("The results mix different n; group by n before comparing them.")
    if not np.allclose(df["theoretical"], REFERENCE, rtol=0, atol=TOLERANCE):
        raise ValueError(f"The 'theoretical' column differs from {REFERENCE}.")
    if not np.allclose(df["abs_error"], (df["estimate"] - REFERENCE).abs(), rtol=0, atol=TOLERANCE):
        raise ValueError("The 'abs_error' column is inconsistent with the estimates.")

    # Every method must have exactly the expected budgets, and trial IDs 0..T-1 once each.
    for method, group in df.groupby("method"):
        found = [int(s) for s in sorted(group["shots"].unique())]
        if found != EXPECTED_SHOTS:
            raise ValueError(f"{method}: shot budgets {found} differ from the expected {EXPECTED_SHOTS}.")
        for shots, g in group.groupby("shots"):
            if sorted(g["trial"]) != list(range(EXPECTED_TRIALS)):
                raise ValueError(f"{method} at {shots} shots: expected trial IDs 0..{EXPECTED_TRIALS - 1} "
                                 f"exactly once each, found {len(g)} rows.")

    print("Validation passed.")
    print(f"Theoretical overlap: {REFERENCE:.4f}")
    print(f"Shot budgets: {EXPECTED_SHOTS}   Trials per budget: {EXPECTED_TRIALS}")
    return df


# ---------------------------------------------------------------- summary statistics
def summarize(df):
    """Per (method, shots): mean estimate, spread, absolute error, RMSE."""
    rows = []
    for (method, shots), g in df.groupby(["method", "shots"]):
        est = g["estimate"].to_numpy()
        err = np.abs(est - REFERENCE)
        rows.append(dict(
            method=method, shots=shots, trials=len(g),
            mean_estimate=est.mean(),
            std=est.std(ddof=1),
            mean_abs_error=err.mean(),
            mae_sem=err.std(ddof=1) / np.sqrt(len(err)),      # uncertainty of the mean |err|
            rmse=np.sqrt(np.mean((est - REFERENCE) ** 2)),
            abs_error_of_mean=abs(est.mean() - REFERENCE),
        ))
    return pd.DataFrame(rows)


def wide_table(summary):
    """Side-by-side table, one row per shot budget."""
    swap = summary[summary.method == "destructive_swap"].set_index("shots").sort_index()
    shad = summary[summary.method == "classical_shadows"].set_index("shots").sort_index()
    t = pd.DataFrame({
        "Shots": swap.index,
        "Theoretical": REFERENCE,
        "SWAP mean est.": swap["mean_estimate"].values,
        "Shadows mean est.": shad["mean_estimate"].values,
        "SWAP mean |err|": swap["mean_abs_error"].values,
        "Shadows mean |err|": shad["mean_abs_error"].values,
        "SWAP RMSE": swap["rmse"].values,
        "Shadows RMSE": shad["rmse"].values,
    })
    # Ratio above 1 means Shadows had the larger observed mean |err|.
    # A zero SWAP error gives NaN (not a tie, not infinity).
    ratio = t["Shadows mean |err|"] / t["SWAP mean |err|"].replace(0, np.nan)
    lower = np.where(t["SWAP mean |err|"] < t["Shadows mean |err|"], "SWAP", "Shadows")
    # "~tie" is a display heuristic (observed errors within 10%), NOT a statistical test.
    t["Lower error"] = np.where(ratio.between(1 - TIE_BAND, 1 + TIE_BAND), "~tie", lower)
    t["Shadows/SWAP error"] = ratio
    return t


# ---------------------------------------------------------------- figures and tables
def make_plots(df, summary, table, trials):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    shots = np.array(EXPECTED_SHOTS)

    def per_method(method):
        return summary[summary.method == method].sort_values("shots")

    # (a) mean absolute error vs shots; the dashed line is an illustrative 1/sqrt(N) guide, not a fit
    for m in NAMES:
        g = per_method(m)
        ax[0].errorbar(g["shots"], g["mean_abs_error"], yerr=g["mae_sem"], marker="o", capsize=3,
                       color=COLORS[m], label=NAMES[m])
    guide = table["SWAP mean |err|"].iloc[0] * np.sqrt(shots[0] / shots)
    ax[0].plot(shots, guide, "k--", lw=1, alpha=0.5, label=r"$1/\sqrt{N}$ guide (illustrative)")
    ax[0].set(xscale="log", yscale="log", xlabel="Total shots", ylabel="Mean absolute error",
              title="(a) Absolute error vs shots")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    # (b) mean estimate +/- 1 std, with the theoretical value
    for m, offset in zip(NAMES, (0.97, 1.03)):
        g = per_method(m)
        ax[1].errorbar(g["shots"] * offset, g["mean_estimate"], yerr=g["std"], marker="o", capsize=3,
                       color=COLORS[m], label=NAMES[m])
    ax[1].axhline(REFERENCE, color="black", ls="--", lw=1.2, label=f"Theoretical = {REFERENCE:.2f}")
    ax[1].set(xscale="log", xlabel="Total shots", ylabel="Estimated overlap",
              title=f"(b) Mean estimate \u00b1 1 std ({trials} trials)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # (c) every individual trial at two budgets
    positions, data, colors = [], [], []
    for i, budget in enumerate([1000, 10000]):
        for j, m in enumerate(NAMES):
            positions.append(i * 3 + j)
            data.append(df[(df.method == m) & (df.shots == budget)]["estimate"].to_numpy())
            colors.append(COLORS[m])
    box = ax[2].boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(0)
    for patch, color, pos, values in zip(box["boxes"], colors, positions, data):
        patch.set(facecolor=color, alpha=0.35)
        ax[2].scatter(pos + rng.uniform(-0.18, 0.18, len(values)), values, s=12, color=color, zorder=3)
    ax[2].axhline(REFERENCE, color="black", ls="--", lw=1.2)
    ax[2].set_xticks([0.5, 3.5])
    ax[2].set_xticklabels(["1000 shots", "10000 shots"])
    ax[2].set(ylabel="Estimated overlap", title="(c) Spread of individual trials")
    ax[2].grid(alpha=0.3, axis="y")
    ax[2].legend(handles=[Patch(color=COLORS[m], alpha=0.5, label=NAMES[m]) for m in NAMES])

    fig.suptitle(f"SWAP or Shadows? Overlap of State A and State B "
                 f"(theoretical {REFERENCE:.2f}, ideal Aer simulation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "comparison_plots.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_table_image(table, trials):
    cols = ["Shots", "SWAP mean est.", "Shadows mean est.", "SWAP mean |err|", "Shadows mean |err|",
            "Lower error"]
    cells = [[f"{int(r['Shots'])}", f"{r['SWAP mean est.']:.4f}", f"{r['Shadows mean est.']:.4f}",
              f"{r['SWAP mean |err|']:.4f}", f"{r['Shadows mean |err|']:.4f}", r["Lower error"]]
             for _, r in table.iterrows()]
    fig, ax = plt.subplots(figsize=(13, 0.65 * len(cells) + 1.5))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.15, 1.6)
    for j in range(len(cols)):
        tb[0, j].set_facecolor("#333333")
        tb[0, j].set_text_props(color="white", weight="bold")
    ax.set_title(f"Theoretical overlap = {REFERENCE:.2f}  |  means over {trials} trials per budget",
                 fontsize=11, pad=15)
    fig.savefig(RES / "headline_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_markdown(table):
    lines = ["| Shots | Theoretical | SWAP mean est. | Shadows mean est. | "
             "SWAP mean abs err | Shadows mean abs err | Lower error |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for _, r in table.iterrows():
        lines.append(f"| {int(r['Shots'])} | {r['Theoretical']:.2f} | {r['SWAP mean est.']:.4f} | "
                     f"{r['Shadows mean est.']:.4f} | {r['SWAP mean |err|']:.4f} | "
                     f"{r['Shadows mean |err|']:.4f} | {r['Lower error']} |")
    (RES / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = load_and_check()
    summary = summarize(df)
    table = wide_table(summary)

    table.to_csv(RES / "comparison_table.csv", index=False)
    write_markdown(table)
    make_plots(df, summary, table, EXPECTED_TRIALS)
    make_table_image(table, EXPECTED_TRIALS)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\nComparison summary:")
    print(table.round(4).to_string(index=False))
    print("\nAll comparison outputs were written to:", RES.resolve())


if __name__ == "__main__":
    main()