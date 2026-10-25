"""
step5_conventions.py - confirms the layout / bit-order claims in HANDOFF.md.
Uses only the ideal Aer simulator and states.py.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from states import N_QUBITS, THETA_A, THETA_B, prepare_joint

sim = AerSimulator(method="statevector")   # ideal, no noise
n = N_QUBITS
shots = 100_000


def run(qc, shots=shots):
    qc = qc.copy()
    qc.measure_all()
    return sim.run(transpile(qc, sim), shots=shots).result().get_counts()


# ---- Check A: little-endian bit order --------------------------------------
print("CHECK A: bit order (X gate on qubit 0 only, 4 qubits)")
qc = QuantumCircuit(2 * n)
qc.x(0)
print("  counts key =", list(run(qc, shots=10).keys())[0], " (expected '0001': qubit 0 is the RIGHTMOST bit)")

# ---- Check B: register layout of prepare_joint() ----------------------------
print("\nCHECK B: prepare_joint() layout, P(qubit = 1) for each qubit")
counts = run(prepare_joint())
p1 = [sum(c for k, c in counts.items() if k[::-1][q] == "1") / shots for q in range(2 * n)]
exp_a = np.sin(THETA_A / 2) ** 2
exp_b = np.sin(THETA_B / 2) ** 2
for q in range(2 * n):
    exp = exp_a if q < n else exp_b
    reg = "A" if q < n else "B"
    print(f"  qubit {q} (register {reg}): measured {p1[q]:.4f}   expected {exp:.4f}")