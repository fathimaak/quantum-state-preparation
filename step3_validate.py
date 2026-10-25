"""
step3_validate.py - validates states.py using an ideal Qiskit Aer simulation.
"""
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from states import (N_QUBITS, THETA_A, THETA_B, DELTA,
                    prepare_state, theoretical_overlap)

sim = AerSimulator(method="statevector")   # ideal: no noise
n = N_QUBITS


def aer_statevector(qc):
    """Run a circuit on ideal Aer and return the exact final state."""
    qc = qc.copy()
    qc.save_statevector()
    result = sim.run(transpile(qc, sim)).result()
    return Statevector(result.get_statevector())


# ---- Check 1: the formula ---------------------------------------------------
print("CHECK 1: theoretical overlap from the formula")
print(f"  cos(delta/2)^(2n) = {theoretical_overlap():.6f}")

# ---- Check 2: overlap computed from Aer statevectors ------------------------
print("\nCHECK 2: overlap from ideal Aer statevectors")
sv_a = aer_statevector(prepare_state(n, THETA_A))
sv_b = aer_statevector(prepare_state(n, THETA_B))
inner = sv_a.inner(sv_b)
print("  State A amplitudes:", np.round(sv_a.data.real, 6))
print("  State B amplitudes:", np.round(sv_b.data.real, 6))
print(f"  <A|B>     = {inner.real:.6f}")
print(f"  |<A|B>|^2 = {abs(inner) ** 2:.6f}")

# ---- Check 3: compute-uncompute test with shots -----------------------------
# Prepare A, then undo B. P(all zeros) = |<B|A>|^2. This is only a sanity check
# on the preparation circuits; it is NOT one of the two methods being compared.
print("\nCHECK 3: compute-uncompute test (100000 shots)")
shots = 100_000
qc = prepare_state(n, THETA_A)
qc.compose(prepare_state(n, THETA_B).inverse(), inplace=True)
qc.measure_all()
counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
p_zero = counts.get("0" * n, 0) / shots
print(f"  counts = {counts}")
print(f"  P(all zeros) = {p_zero:.4f}   (expected {theoretical_overlap():.4f}, +/- about 0.004)")

# ---- Check 4: formula vs Aer over several delta values ----------------------
print("\nCHECK 4: formula vs Aer for several delta values (n = %d)" % n)
print("  delta/pi   formula    Aer")
for k in [0, 1 / 3, 1 / 2, 2 / 3, 1]:
    d = k * np.pi
    a = aer_statevector(prepare_state(n, THETA_A))
    b = aer_statevector(prepare_state(n, THETA_A + d))
    print(f"  {k:6.3f}   {theoretical_overlap(n, d):.6f}   {abs(a.inner(b)) ** 2:.6f}")