"""
states.py - Quantum State Preparation module (owner: Fathima)

Single source of truth for State A, State B and the theoretical overlap.
Everyone on the team imports from this file; nobody redefines the states.

Overlap convention used by the whole project:
    overlap = Tr(rho sigma) = |<A|B>|^2     (squared inner product)
"""
import numpy as np
from qiskit import QuantumCircuit

# ---- Phase 1 parameters (change ONLY here) --------------------------------
N_QUBITS = 2                  # qubits per register
THETA_A = np.pi / 3           # State A: every qubit gets Ry(THETA_A)
DELTA = np.pi / 2             # difference in rotation angle between B and A
THETA_B = THETA_A + DELTA     # State B: every qubit gets Ry(THETA_B)
# ----------------------------------------------------------------------------


def prepare_state(n, theta):
    """n-qubit circuit that prepares the product state (Ry(theta)|0>) on every qubit."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(theta, i)
    return qc


def prepare_joint(n=N_QUBITS, theta_a=THETA_A, theta_b=THETA_B):
    """2n-qubit circuit with State A on qubits 0..n-1 and State B on qubits n..2n-1.

    Pairing convention: pair i is (A_i, B_i) = (qubit i, qubit n+i).
    No measurements are added; teammates append their own.
    """
    qc = QuantumCircuit(2 * n)
    qc.compose(prepare_state(n, theta_a), qubits=range(0, n), inplace=True)
    qc.compose(prepare_state(n, theta_b), qubits=range(n, 2 * n), inplace=True)
    return qc


def theoretical_overlap(n=N_QUBITS, delta=DELTA):
    """Closed-form overlap |<A|B>|^2 = cos(delta/2)^(2n) for the product family."""
    return float(np.cos(delta / 2) ** (2 * n))


if __name__ == "__main__":
    print(f"n = {N_QUBITS} qubits per register")
    print(f"theta_A = {THETA_A:.6f} rad   theta_B = {THETA_B:.6f} rad   delta = {DELTA:.6f} rad")
    print(f"Theoretical overlap = {theoretical_overlap():.6f}")
    print()
    print(prepare_joint().draw())