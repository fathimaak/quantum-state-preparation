"""
shadows_demo.py - ONE small run of the classical-shadows method with a readable printout.
Run:  python shadows_demo.py
"""
import numpy as np
from states import N_QUBITS, THETA_A, THETA_B, theoretical_overlap
from shadows import (collect_snapshots_aer, estimate_overlap_from_snapshots, AXES)

SHOTS = 2000
SEED = 7
n = N_QUBITS

print(f"States: n = {n} qubits per register, theta_A = {THETA_A:.4f}, theta_B = {THETA_B:.4f}")
print(f"Theoretical overlap (reference) = {theoretical_overlap():.4f}\n")

bases, signs = collect_snapshots_aer(n, THETA_A, THETA_B, shots=SHOTS, seed=SEED)
print(f"Collected {len(bases)} snapshots. First 5 (axis, sign) for qubits 0..{2*n-1}:")
for j in range(5):
    cells = [f"{AXES[a]}{'+' if s > 0 else '-'}" for a, s in zip(bases[j], signs[j])]
    print("  shot", j, ":", "  ".join(cells), "   (qubits 0,1 = register A; 2,3 = register B)")

est, se = estimate_overlap_from_snapshots(bases, signs, n)
theo = theoretical_overlap()
print(f"\nEstimated overlap = {est:.4f} +/- {se:.4f}   (1 standard error)")
print(f"Theoretical       = {theo:.4f}")
print(f"Absolute error    = {abs(est - theo):.4f}   ({abs(est - theo) / se:.2f} standard errors)")
