"""
shadows_experiment.py - accuracy-versus-shots sweep for the classical-shadows method.

Writes two CSV files in ./results/ for Donita:
  shadows_results.csv  one row per run: method,n,shots,trial,seed,estimate,stderr,theoretical,abs_error
  shadows_summary.csv  one row per shot budget: mean estimate, mean abs error, RMSE, empirical std

Run:   python shadows_experiment.py                 (Aer, 20 trials per budget, about 2 minutes)
       python shadows_experiment.py --trials 50     (smoother curves, slower)
       python shadows_experiment.py --sampler exact (Qiskit-free, very fast, validation only)
"""
import argparse
import os
import numpy as np
import pandas as pd
from states import N_QUBITS, THETA_A, THETA_B, theoretical_overlap
from shadows import estimate_overlap_shadows

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--sampler", choices=["aer", "exact"], default="aer")
parser.add_argument("--shots", type=int, nargs="+", default=[100, 200, 500, 1000, 2000, 5000, 10000])
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
rows = []
for shots in args.shots:
    for trial in range(args.trials):
        seed = 1000 * shots + trial                      # fixed, so every run is reproducible
        r = estimate_overlap_shadows(N_QUBITS, THETA_A, THETA_B, shots, seed, args.sampler)
        r.update(trial=trial, seed=seed)
        rows.append(r)
    print(f"finished {shots} shots x {args.trials} trials", flush=True)

df = pd.DataFrame(rows)[["method", "n", "shots", "trial", "seed", "estimate", "stderr",
                         "theoretical", "abs_error"]]
df.to_csv("results/shadows_results.csv", index=False)

theo = theoretical_overlap()
summary = df.groupby("shots").agg(mean_estimate=("estimate", "mean"),
                                  empirical_std=("estimate", "std"),
                                  mean_abs_error=("abs_error", "mean"),
                                  mean_reported_se=("stderr", "mean")).reset_index()
summary["rmse"] = [float(np.sqrt(np.mean((g.estimate - theo) ** 2))) for _, g in df.groupby("shots")]
summary.to_csv("results/shadows_summary.csv", index=False)

pd.set_option("display.width", 160)
print("\nSummary (theoretical overlap = %.4f):" % theo)
print(summary.round(4).to_string(index=False))
slope = np.polyfit(np.log(summary.shots), np.log(summary.rmse), 1)[0]
print(f"\nlog-log slope of RMSE versus shots = {slope:.2f}   (statistical noise predicts about -0.5)")
