"""
destructive_swap.py - Destructive SWAP test overlap estimator (owner: Sankhana)

Estimates   overlap = Tr(rho sigma) = |<A|B>|^2   for the two states defined in states.py,
using transversal Bell-basis measurements (CNOT + Hadamard) on corresponding qubit pairs.

Conventions (identical to HANDOFF.md):
  * Register A = qubits 0..n-1, register B = qubits n..2n-1 (from prepare_joint()).
  * Pair i is (A_i, B_i) = (qubit i, qubit n+i).
  * Qiskit bitstrings are little-endian: in a counts string, qubit 0 is the RIGHTMOST character.
  * For pair i, CNOT(A_i -> B_i) followed by H(A_i) maps the antisymmetric singlet |Psi->
    to outcome A_i=1, B_i=1.
  * The pair SWAP expectation value is <S_i> = 1 - 2 P(11).
  * For product states, the global SWAP operator is S = prod_i S_i.
    Each shot evaluates to (-1)^k where k is the number of pairs with outcome (1, 1).
    Averaging across shots yields the unbiased overlap estimate Tr(rho sigma).
"""
import argparse
import math
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from states import N_QUBITS, THETA_A, THETA_B, DELTA, prepare_joint, theoretical_overlap


def build_destructive_swap_circuit(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B):
    """Build state-preparation + transversal Bell measurement circuit.

    Pair i is (A_i, B_i) = (qubit i, qubit n+i).
    The CNOT(A_i -> B_i), H(A_i) Bell-measurement mapping makes
    outcome A_i=B_i=1 the antisymmetric |Psi-> outcome.
    """
    qc = prepare_joint(n, theta_a, theta_b)
    qc.barrier()

    for i in range(n):
        a = i
        b = n + i
        qc.cx(a, b)
        qc.h(a)

    qc.measure_all()
    return qc


def bit_for_qubit(bitstring, qubit_index):
    """Read a qubit's measured bit from Qiskit's displayed count key.

    Qiskit displays the highest-index classical bit on the left (little-endian).
    """
    compact = bitstring.replace(" ", "")
    return int(compact[-1 - qubit_index])


def estimate_from_counts(counts, n_qubits_per_register=N_QUBITS):
    """Return pair P(11) probabilities, pair estimates, global overlap estimate, and stderr.

    Returns:
        pair_p11: list of float P(A_i=1, B_i=1) per pair
        pair_estimates: list of float 1 - 2*P(11) per pair
        estimate: float global overlap estimate
        standard_error: float standard error of the mean
    """
    shots = sum(counts.values())
    if shots == 0:
        return [0.0] * n_qubits_per_register, [1.0] * n_qubits_per_register, 0.0, 0.0

    pair_p11 = []
    pair_estimates = []

    for i in range(n_qubits_per_register):
        a, b = i, n_qubits_per_register + i
        p11_count = sum(
            count for bits, count in counts.items()
            if bit_for_qubit(bits, a) == 1 and bit_for_qubit(bits, b) == 1
        )
        p11 = p11_count / shots
        pair_p11.append(p11)
        pair_estimates.append(1.0 - 2.0 * p11)

    # A single shot's global SWAP eigenvalue is the product of pair eigenvalues:
    # (+1) if an even number of pairs have (1, 1), (-1) if an odd number.
    pos_counts = 0
    neg_counts = 0
    for bits, count in counts.items():
        value = 1
        for i in range(n_qubits_per_register):
            if bit_for_qubit(bits, i) == 1 and bit_for_qubit(bits, n_qubits_per_register + i) == 1:
                value *= -1
        if value == 1:
            pos_counts += count
        else:
            neg_counts += count

    estimate = float((pos_counts - neg_counts) / shots)
    if shots > 1:
        # Sample variance for binary {+1, -1} outcomes is (1 - estimate^2) * (shots / (shots - 1))
        # Standard error of the mean = sqrt(var / shots) = sqrt((1 - estimate^2) / (shots - 1))
        var_sample = max(0.0, (1.0 - estimate ** 2) / (shots - 1))
        standard_error = float(math.sqrt(var_sample))
    else:
        standard_error = 0.0

    return pair_p11, pair_estimates, estimate, standard_error


def collect_counts_aer(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000, seed=None,
                       backend=None):
    """Execute destructive SWAP circuit on Qiskit AerSimulator."""
    sim = backend or AerSimulator(method="statevector")
    qc = build_destructive_swap_circuit(n, theta_a, theta_b)
    tqc = transpile(qc, sim)
    run_kwargs = {"shots": shots}
    if seed is not None:
        run_kwargs["seed_simulator"] = int(seed)
    result = sim.run(tqc, **run_kwargs).result()
    return result.get_counts()


def collect_counts_exact(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000, seed=None):
    """Qiskit-FREE exact sampler for validation and rapid testing.

    For product states, each pair (A_i, B_i) is independent with exact probabilities:
        P(00) = 0.5 * cos^2(delta/2)
        P(10) = 0.5 * cos^2((theta_a + theta_b)/2)
        P(01) = 0.5 * sin^2((theta_a + theta_b)/2)
        P(11) = 0.5 * sin^2(delta/2)
    where delta = theta_b - theta_a.
    """
    rng = np.random.default_rng(seed)
    delta = theta_b - theta_a
    p00 = 0.5 * (np.cos(delta / 2.0) ** 2)
    p10 = 0.5 * (np.cos((theta_a + theta_b) / 2.0) ** 2)
    p01 = 0.5 * (np.sin((theta_a + theta_b) / 2.0) ** 2)
    p11 = 0.5 * (np.sin(delta / 2.0) ** 2)
    pair_probs = np.array([p00, p01, p10, p11])
    pair_probs = pair_probs / np.sum(pair_probs)

    # Sample each pair independently
    # outcomes 0..3 correspond to: 0 -> a=0,b=0; 1 -> a=0,b=1; 2 -> a=1,b=0; 3 -> a=1,b=1
    pair_choices = rng.choice(4, size=(shots, n), p=pair_probs)

    counts = {}
    for shot in range(shots):
        bits = [0] * (2 * n)
        for i in range(n):
            choice = pair_choices[shot, i]
            a_bit = 1 if choice in (2, 3) else 0
            b_bit = 1 if choice in (1, 3) else 0
            bits[i] = a_bit
            bits[n + i] = b_bit
        # Qiskit format: bit string with qubit 2n-1 on left and qubit 0 on right
        bitstr = "".join(str(b) for b in reversed(bits))
        counts[bitstr] = counts.get(bitstr, 0) + 1

    return counts


def estimate_overlap_destructive_swap(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B, shots=1000,
                                      seed=None, sampler="aer", backend=None):
    """ONE-CALL INTERFACE for Donita / the whole team.

    `shots` = total number of measurement shots.
    sampler = "aer" (Qiskit Aer simulator, default) or "exact" (fast validation sampler).
    Returns a dict:
        method: "destructive_swap"
        n: int
        shots: int
        estimate: float
        stderr: float
        theoretical: float
        abs_error: float
    """
    if sampler == "aer":
        counts = collect_counts_aer(n, theta_a, theta_b, shots, seed=seed, backend=backend)
    elif sampler == "exact":
        counts = collect_counts_exact(n, theta_a, theta_b, shots, seed=seed)
    else:
        raise ValueError("sampler must be 'aer' or 'exact'")

    _, _, est, se = estimate_from_counts(counts, n)
    theo = theoretical_overlap(n, theta_b - theta_a)
    return dict(
        method="destructive_swap",
        n=n,
        shots=shots,
        estimate=est,
        stderr=se,
        theoretical=theo,
        abs_error=abs(est - theo),
    )


# Alias for team convenience
estimate_overlap_swap = estimate_overlap_destructive_swap


def main():
    parser = argparse.ArgumentParser(description="Destructive SWAP overlap estimate")
    parser.add_argument("--shots", type=int, default=10000, help="number of simulator shots")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--sampler", choices=["aer", "exact"], default="aer")
    args = parser.parse_args()

    if args.shots < 1:
        parser.error("--shots must be at least 1")

    res = estimate_overlap_destructive_swap(
        n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B,
        shots=args.shots, seed=args.seed, sampler=args.sampler
    )

    print("=" * 60)
    print("DESTRUCTIVE SWAP TEST")
    print("=" * 60)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k:14s}: {v:.6f}")
        else:
            print(f"{k:14s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
