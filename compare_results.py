"""
compare_results.py - Integration and results visualization (owner: Donita)

Reads the detailed CSVs from Sankhana (destructive SWAP) and Haneen (classical
shadows), checks that they are comparable, and produces:
    results/comparison_table.csv      per-budget table, both methods side by side
    results/comparison_table.md       same table, ready to paste into the report
    results/comparison_plots.png      error-vs-shots and estimate-vs-shots figure
    results/headline_table.png        compact table image for slides

Run:  python compare_results.py
Needs only pandas, numpy, matplotlib.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES = HERE / "results"
SWAP_CSV = RES / "destructive_swap_results.csv"
SHAD_CSV = RES / "shadows_results.csv"
REFERENCE = 0.25            # theoretical overlap from states.py (Fathima)
NAMES = {"destructive_swap": "Destructive SWAP", "classical_shadows": "Classical Shadows"}
COLORS = {"destructive_swap": "#1f77b4", "classical_shadows": "#d95f02"}


def load_and_check():
    df = pd.concat([pd.read_csv(SWAP_CSV), pd.read_csv(SHAD_CSV)], ignore_index=True)
    # 1) both methods use the same reference value, and it matches the handoff
    assert np.allclose(df["theoretical"], REFERENCE, atol=1e-9), "theoretical value differs from 0.25"
    # 2) same shot budgets and same number of trials per budget (equal-cost comparison)
    counts = df.groupby(["method", "shots"]).size().unstack(0)
    assert counts.notna().all().all() and (counts.nunique(axis=1) == 1).all(), "budgets/trials differ"
    # 3) abs_error column is consistent with estimate
    assert np.allclose(df["abs_error"], (df["estimate"] - REFERENCE).abs(), atol=1e-9)
    return df


def summarize(df):
    """Per (method, shots): mean estimate, spread, absolute error, rmse."""
    rows = []
    for (m, s), g in df.groupby(["method", "shots"]):
        est = g["estimate"].to_numpy()
        ae = (est - REFERENCE).__abs__()
        rows.append(dict(
            method=m, shots=s, trials=len(g),
            mean_estimate=est.mean(),
            std=est.std(ddof=1),
            mean_abs_error=ae.mean(),
            mae_sem=ae.std(ddof=1) / np.sqrt(len(ae)),   # uncertainty of the mean abs error
            rmse=np.sqrt(((est - REFERENCE) ** 2).mean()),
            abs_error_of_mean=abs(est.mean() - REFERENCE),  # bias-type error of the averaged estimate
        ))
    return pd.DataFrame(rows)


def wide_table(s):
    """Side-by-side table, one row per shot budget."""
    sw = s[s.method == "destructive_swap"].set_index("shots")
    sh = s[s.method == "classical_shadows"].set_index("shots")
    t = pd.DataFrame({
        "Shots": sw.index,
        "Theoretical": REFERENCE,
        "SWAP mean est.": sw["mean_estimate"].values,
        "Shadows mean est.": sh["mean_estimate"].values,
        "SWAP mean |err|": sw["mean_abs_error"].values,
        "Shadows mean |err|": sh["mean_abs_error"].values,
        "SWAP RMSE": sw["rmse"].values,
        "Shadows RMSE": sh["rmse"].values,
    })
    better = np.where(t["SWAP mean |err|"] < t["Shadows mean |err|"], "SWAP", "Shadows")
    ratio = t["Shadows mean |err|"] / t["SWAP mean |err|"]
    # call it a tie if the two are within 10% of each other
    t["Lower error"] = np.where((ratio > 0.9) & (ratio < 1.1), "~tie", better)
    t["Shadows/SWAP error"] = ratio
    return t


def make_plots(df, s, t):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    shots = np.array(sorted(df["shots"].unique()))

    # (a) mean absolute error vs shots (log-log) with 1/sqrt(N) guide
    for m in NAMES:
        g = s[s.method == m].sort_values("shots")
        ax[0].errorbar(g["shots"], g["mean_abs_error"], yerr=g["mae_sem"], marker="o", capsize=3,
                       color=COLORS[m], label=NAMES[m])
    ref_line = t["SWAP mean |err|"].iloc[0] * np.sqrt(shots[0] / shots)
    ax[0].plot(shots, ref_line, "k--", lw=1, alpha=0.5, label=r"$1/\sqrt{N}$ guide")
    ax[0].set(xscale="log", yscale="log", xlabel="Total shots", ylabel="Mean absolute error",
              title="(a) Absolute error vs shots")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    # (b) mean estimate +/- 1 std vs theoretical
    for m, off in zip(NAMES, (0.97, 1.03)):
        g = s[s.method == m].sort_values("shots")
        ax[1].errorbar(g["shots"] * off, g["mean_estimate"], yerr=g["std"], marker="o", capsize=3,
                       color=COLORS[m], label=NAMES[m])
    ax[1].axhline(REFERENCE, color="k", ls="--", lw=1.2, label="Theoretical = 0.25")
    ax[1].set(xscale="log", xlabel="Total shots", ylabel="Estimated overlap",
              title="(b) Mean estimate ± 1 std (20 trials)")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    # (c) every individual trial at 1000 and 10000 shots (box + points)
    positions, data, cols, labels = [], [], [], []
    for i, sh in enumerate([1000, 10000]):
        for j, m in enumerate(NAMES):
            positions.append(i * 3 + j)
            data.append(df[(df.method == m) & (df.shots == sh)]["estimate"].to_numpy())
            cols.append(COLORS[m])
        labels.append(f"{sh} shots")
    bp = ax[2].boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(0)
    for patch, c, p, d in zip(bp["boxes"], cols, positions, data):
        patch.set(facecolor=c, alpha=0.35)
        ax[2].scatter(p + rng.uniform(-0.18, 0.18, len(d)), d, s=12, color=c, zorder=3)
    ax[2].axhline(REFERENCE, color="k", ls="--", lw=1.2)
    ax[2].set_xticks([0.5, 3.5]); ax[2].set_xticklabels(labels)
    ax[2].set(ylabel="Estimated overlap", title="(c) Spread of individual trials")
    ax[2].grid(alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(color=COLORS[m], alpha=0.5, label=NAMES[m]) for m in NAMES])

    fig.suptitle("SWAP or Shadows? Overlap of State A and State B (theoretical 0.25, ideal Aer simulation)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "comparison_plots.png", dpi=170)
    plt.close(fig)


def make_table_image(t):
    cols = ["Shots", "SWAP mean est.", "Shadows mean est.", "SWAP mean |err|", "Shadows mean |err|", "Lower error"]
    cell = [[f"{r['Shots']}", f"{r['SWAP mean est.']:.4f}", f"{r['Shadows mean est.']:.4f}",
             f"{r['SWAP mean |err|']:.4f}", f"{r['Shadows mean |err|']:.4f}", r["Lower error"]]
            for _, r in t.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(cell) + 1.2))
    ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.5)
    for j in range(len(cols)):
        tb[0, j].set_facecolor("#333333"); tb[0, j].set_text_props(color="white", weight="bold")
    ax.set_title("Theoretical overlap = 0.25  |  means over 20 trials per budget", fontsize=10)
    fig.savefig(RES / "headline_table.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_markdown(t):
    lines = ["| Shots | Theoretical | SWAP mean est. | Shadows mean est. | SWAP mean abs err | Shadows mean abs err | Lower error |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['Shots']} | {r['Theoretical']:.2f} | {r['SWAP mean est.']:.4f} | "
                     f"{r['Shadows mean est.']:.4f} | {r['SWAP mean |err|']:.4f} | "
                     f"{r['Shadows mean |err|']:.4f} | {r['Lower error']} |")
    (RES / "comparison_table.md").write_text("\n".join(lines) + "\n")


def main():
    df = load_and_check()
    s = summarize(df)
    t = wide_table(s)
    t.to_csv(RES / "comparison_table.csv", index=False)
    write_markdown(t)
    make_plots(df, s, t)
    make_table_image(t)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print("\nOutputs written to", RES)


if __name__ == "__main__":
    main()
