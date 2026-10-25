"""
destructive_swap_experiment.py - accuracy-versus-shots sweep for the destructive-SWAP method.

Writes CSV files in ./results/ for Donita:
  destructive_swap_results.csv
      one row per run: method,n,shots,trial,seed,estimate,stderr,theoretical,abs_error
  destructive_swap_summary.csv
      one row per shot budget: mean estimate, empirical std, mean abs error, mean reported se, RMSE

Run:   python destructive_swap_experiment.py                 (Aer, 20 trials per budget)
       python destructive_swap_experiment.py --trials 50     (smoother curves)
       python destructive_swap_experiment.py --sampler exact (Qiskit-free, very fast validation)
"""
import argparse
import csv
import os
import numpy as np

from states import N_QUBITS, THETA_A, THETA_B, theoretical_overlap
from destructive_swap import estimate_overlap_destructive_swap


def run_experiment():
    parser = argparse.ArgumentParser(description="Destructive SWAP accuracy vs shots sweep")
    parser.add_argument("--trials", type=int, default=20, help="number of trials per shot budget")
    parser.add_argument("--sampler", choices=["aer", "exact"], default="aer")
    parser.add_argument("--shots", type=int, nargs="+",
                        default=[100, 200, 500, 1000, 2000, 5000, 10000])
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    rows = []
    fieldnames = ["method", "n", "shots", "trial", "seed", "estimate", "stderr",
                  "theoretical", "abs_error"]

    for shots in args.shots:
        for trial in range(args.trials):
            seed = 1000 * shots + trial  # reproducible seed convention
            r = estimate_overlap_destructive_swap(
                n=N_QUBITS,
                theta_a=THETA_A,
                theta_b=THETA_B,
                shots=shots,
                seed=seed,
                sampler=args.sampler,
            )
            r["trial"] = trial
            r["seed"] = seed
            rows.append(r)
        print(f"finished {shots} shots x {args.trials} trials", flush=True)

    # Write detailed results CSV
    results_filename = "results/destructive_swap_results.csv"
    with open(results_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    theo = theoretical_overlap()

    # Aggregate summary metrics per shot budget
    summary_rows = []
    for shots in args.shots:
        group = [r for r in rows if r["shots"] == shots]
        estimates = np.array([r["estimate"] for r in group], dtype=float)
        abs_errors = np.array([r["abs_error"] for r in group], dtype=float)
        stderrs = np.array([r["stderr"] for r in group], dtype=float)

        mean_est = float(np.mean(estimates))
        emp_std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
        mean_mae = float(np.mean(abs_errors))
        mean_se = float(np.mean(stderrs))
        rmse = float(np.sqrt(np.mean((estimates - theo) ** 2)))

        summary_rows.append({
            "shots": shots,
            "mean_estimate": mean_est,
            "empirical_std": emp_std,
            "mean_abs_error": mean_mae,
            "mean_reported_se": mean_se,
            "rmse": rmse,
        })

    # Write summary CSV
    summary_fieldnames = ["shots", "mean_estimate", "empirical_std", "mean_abs_error",
                          "mean_reported_se", "rmse"]
    summary_filename = "results/destructive_swap_summary.csv"
    with open(summary_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for s in summary_rows:
            writer.writerow(s)

    # Print summary table matching pandas formatting
    print(f"\nSummary (theoretical overlap = {theo:.4f}):")
    header_str = f"{'shots':>7}  {'mean_estimate':>13}  {'empirical_std':>13}  {'mean_abs_error':>14}  {'mean_reported_se':>16}  {'rmse':>8}"
    print(header_str)
    for s in summary_rows:
        print(
            f"{s['shots']:7d}  "
            f"{s['mean_estimate']:13.4f}  "
            f"{s['empirical_std']:13.4f}  "
            f"{s['mean_abs_error']:14.4f}  "
            f"{s['mean_reported_se']:16.4f}  "
            f"{s['rmse']:8.4f}"
        )

    # Calculate log-log slope of RMSE versus shots
    shots_arr = np.array([s["shots"] for s in summary_rows], dtype=float)
    rmse_arr = np.array([s["rmse"] for s in summary_rows], dtype=float)
    slope = np.polyfit(np.log(shots_arr), np.log(rmse_arr), 1)[0]
    print(f"\nlog-log slope of RMSE versus shots = {slope:.2f}   (statistical noise predicts about -0.5)")


if __name__ == "__main__":
    run_experiment()
