"""
destructive_swap_demo.py - ONE small run of the destructive SWAP test with a readable printout.
Run:  python destructive_swap_demo.py
"""
from states import N_QUBITS, THETA_A, THETA_B, theoretical_overlap
from destructive_swap import (
    build_destructive_swap_circuit,
    collect_counts_aer,
    estimate_from_counts,
    estimate_overlap_destructive_swap,
)

SHOTS = 2000
SEED = 7
n = N_QUBITS

print("=" * 60)
print("DESTRUCTIVE SWAP TEST DEMO (Sankhana)")
print("=" * 60)
print(f"States: n = {n} qubits per register, theta_A = {THETA_A:.4f}, theta_B = {THETA_B:.4f}")
print(f"Theoretical overlap (reference) = {theoretical_overlap():.4f}\n")

# Run Aer simulation
counts = collect_counts_aer(n, THETA_A, THETA_B, shots=SHOTS, seed=SEED)
p11, pair_estimates, est, se = estimate_from_counts(counts, n)
theo = theoretical_overlap()

print(f"Executed {SHOTS} shots on AerSimulator (seed={SEED}).")
print("Pairwise Bell measurement results:")
for i, (prob, pair_est) in enumerate(zip(p11, pair_estimates)):
    print(f"  Pair {i} (q{i}, q{n+i}): P(11) = {prob:.4f}  ->  <S_{i}> = 1 - 2*P(11) = {pair_est:.4f}")

print(f"\nEstimated overlap = {est:.4f} +/- {se:.4f}   (1 standard error)")
print(f"Theoretical       = {theo:.4f}")
print(f"Absolute error    = {abs(est - theo):.4f}   ({abs(est - theo) / se:.2f} standard errors)")

print("\nSingle-call interface output dictionary:")
out = estimate_overlap_destructive_swap(shots=SHOTS, seed=SEED)
for k, v in out.items():
    print(f"  {k:14s}: {v}")
print("=" * 60)
