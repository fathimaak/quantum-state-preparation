"""
shadows.py - Classical-shadows (randomized-measurement) overlap estimator   (owner: Haneen)

Estimates   overlap = Tr(rho sigma) = |<A|B>|^2   for the two states defined in states.py,
using ONLY random single-qubit measurements. No gate ever connects register A to register B.

Conventions (identical to HANDOFF.md):
  * Register A = qubits 0..n-1, register B = qubits n..2n-1 (joint circuit from prepare_joint()).
  * Qiskit bitstrings are little-endian: in a counts / memory string, qubit 0 is the RIGHTMOST character.

Method in one paragraph
  For every shot we pick, independently for every qubit, a random measurement axis in {X, Y, Z},
  rotate that qubit so the axis becomes Z, measure, and record (axis, sign) with sign = +1 for
  outcome 0 and -1 for outcome 1. For one qubit the "shadow" snapshot is  rho_hat = 3|s><s| - I
  where |s> is the +/-1 eigenstate of the chosen axis. For two snapshots (one of A, one of B):
        Tr(rho_hat sigma_hat) = 9 |<s_A|s_B>|^2 - 4
                              = 1/2 + (9/2) * s_A * s_B * [axis_A == axis_B]          (per qubit)
  A multi-qubit snapshot is a tensor product, so the kernel is the product over qubits:
        K(j,k) = prod_i f_i(j,k)
  and the overlap estimate is the average of K over ALL pairs (A-snapshot j, B-snapshot k):
        O_hat = (1/M^2) * sum_{j,k} K(j,k)
  A and B come from different states, so the j = k terms are valid (nothing to exclude).
"""
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from states import N_QUBITS, THETA_A, THETA_B, prepare_joint, theoretical_overlap

AXES = ("X", "Y", "Z")      # axis index 0, 1, 2


# --------------------------------------------------------------------------- measurement side
def random_bases(shots, n_total, rng):
    """(shots, n_total) array of random axis indices 0=X, 1=Y, 2=Z, uniform and independent."""
    return rng.integers(0, 3, size=(shots, n_total))


def add_basis_rotation(qc, axes_row):
    """Rotate qubit q so that measuring Z afterwards measures the chosen axis.
       X: H          Y: Sdg then H          Z: nothing"""
    for q, a in enumerate(axes_row):
        if a == 0:
            qc.h(q)
        elif a == 1:
            qc.sdg(q)
            qc.h(q)


def parse_memory(bitstring):
    """Qiskit little-endian string -> list of bits where element q is the outcome of qubit q."""
    return [int(ch) for ch in reversed(bitstring.replace(" ", ""))]


def collect_snapshots_aer(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000, seed=None,
                          backend=None):
    """Run `shots` randomized-measurement snapshots on the ideal Aer simulator.

    Shots that happen to share the same random axis pattern are executed together in one circuit
    (so only 3^(2n) distinct circuits are needed), but every shot is still an independent snapshot.
    Returns  bases (shots, 2n) ints in {0,1,2}   and   signs (shots, 2n) ints in {+1,-1}.
    """
    rng = np.random.default_rng(seed)
    n_total = 2 * n
    sim = backend or AerSimulator(method="statevector")
    bases = random_bases(shots, n_total, rng)
    patterns, counts = np.unique(bases, axis=0, return_counts=True)

    circuits = []
    for pat in patterns:
        qc = prepare_joint(n, theta_a, theta_b)      # State A on 0..n-1, State B on n..2n-1
        add_basis_rotation(qc, pat)
        qc.measure_all()
        circuits.append(qc)
    tcircs = transpile(circuits, sim)

    all_bases, all_signs = [], []
    for pat, cnt, tqc in zip(patterns, counts, tcircs):
        mem = sim.run(tqc, shots=int(cnt), memory=True,
                      seed_simulator=int(rng.integers(0, 2**31 - 1))).result().get_memory()
        for s in mem:
            bits = parse_memory(s)
            all_signs.append([1 - 2 * b for b in bits])      # outcome 0 -> +1, outcome 1 -> -1
            all_bases.append(pat)
    all_bases, all_signs = np.array(all_bases), np.array(all_signs)
    order = rng.permutation(len(all_bases))        # shots were grouped by pattern; restore random order
    return all_bases[order], all_signs[order]


def collect_snapshots_exact(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000, seed=None):
    """Qiskit-FREE sampler used only to validate the estimator (fast, thousands of trials).

    Ry(t)|0> has Bloch vector (sin t, 0, cos t), so measuring axis X/Y/Z gives sign +1 with
    probability (1 + r_axis)/2. Qubits are independent (product state)."""
    rng = np.random.default_rng(seed)
    n_total = 2 * n
    thetas = np.array([theta_a] * n + [theta_b] * n)
    bloch = np.stack([np.sin(thetas), np.zeros(n_total), np.cos(thetas)], axis=1)   # (2n, 3)
    bases = random_bases(shots, n_total, rng)
    r = bloch[np.arange(n_total)[None, :], bases]                                    # (shots, 2n)
    p_plus = (1 + r) / 2
    signs = np.where(rng.random((shots, n_total)) < p_plus, 1, -1)
    return bases, signs


# --------------------------------------------------------------------------- estimator side
def estimate_overlap_from_snapshots(bases, signs, n, chunk=500):
    """Cross-correlation estimator of Tr(rho sigma) from A- and B-snapshots.

    Returns (estimate, stderr). The standard error is the leading-order (Hoeffding projection)
    error of a two-sample U-statistic: Var ~ [Var_j(mean_k K) + Var_k(mean_j K)] / M.
    """
    M = bases.shape[0]
    a_ax, a_s = bases[:, :n], signs[:, :n]
    b_ax, b_s = bases[:, n:], signs[:, n:]
    row_mean = np.zeros(M)
    col_sum = np.zeros(M)
    total = 0.0
    for start in range(0, M, chunk):
        stop = min(M, start + chunk)
        K = np.ones((stop - start, M))
        for i in range(n):                                       # product over qubits
            same_axis = (a_ax[start:stop, i][:, None] == b_ax[:, i][None, :])
            sign_prod = a_s[start:stop, i][:, None] * b_s[:, i][None, :]
            K *= 0.5 + 4.5 * same_axis * sign_prod               # 5, -4, or 0.5 per qubit
        row_mean[start:stop] = K.mean(axis=1)
        col_sum += K.sum(axis=0)
        total += K.sum()
    estimate = total / (M * M)
    col_mean = col_sum / M
    stderr = np.sqrt(row_mean.var(ddof=1) / M + col_mean.var(ddof=1) / M)
    return float(estimate), float(stderr)


def estimate_overlap_shadows(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000, seed=None,
                             sampler="aer"):
    """ONE-CALL INTERFACE for Donita / the whole team.

    `shots` = total number of measurement shots (= snapshots of A and of B).
    sampler = "aer" (Qiskit, default) or "exact" (Qiskit-free validation sampler).
    Returns a dict: method, n, shots, estimate, stderr, theoretical, abs_error.
    """
    if sampler == "aer":
        bases, signs = collect_snapshots_aer(n, theta_a, theta_b, shots, seed)
    elif sampler == "exact":
        bases, signs = collect_snapshots_exact(n, theta_a, theta_b, shots, seed)
    else:
        raise ValueError("sampler must be 'aer' or 'exact'")
    est, se = estimate_overlap_from_snapshots(bases, signs, n)
    delta = theta_b - theta_a
    theo = theoretical_overlap(n, delta)
    return dict(method="classical_shadows", n=n, shots=shots, estimate=est, stderr=se,
                theoretical=theo, abs_error=abs(est - theo))


if __name__ == "__main__":
    out = estimate_overlap_shadows(shots=2000, seed=1)
    for k, v in out.items():
        print(f"{k:12s}: {v}")
