"""plot_shadows.py - quick self-check plot: error versus shots, from results/shadows_summary.csv."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = pd.read_csv("results/shadows_summary.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(s.shots, s.rmse, "o-", label="classical shadows (RMSE)")
ref = s.rmse.iloc[0] * np.sqrt(s.shots.iloc[0] / s.shots)
ax.loglog(s.shots, ref, "--", color="gray", label="1/sqrt(shots) reference")
ax.set_xlabel("total shots"); ax.set_ylabel("RMSE of overlap estimate")
ax.set_title("Classical shadows, ideal simulation"); ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.savefig("results/shadows_error_vs_shots.png", dpi=150)
print("saved results/shadows_error_vs_shots.png")
