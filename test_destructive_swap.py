"""
test_destructive_swap.py - unit tests for destructive SWAP test.
Run with:  python -m pytest -v
"""
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from states import N_QUBITS, THETA_A, THETA_B, DELTA, theoretical_overlap
from destructive_swap import (
    build_destructive_swap_circuit,
    bit_for_qubit,
    estimate_from_counts,
    collect_counts_aer,
    collect_counts_exact,
    estimate_overlap_destructive_swap,
    estimate_overlap_swap,
)


def test_reference_value_from_states_py():                                     # T1
    assert theoretical_overlap() == pytest.approx(0.25)


def test_bit_order_is_little_endian():                                         # T2
    # Qiskit count bitstrings have qubit 0 as the RIGHTMOST character
    assert bit_for_qubit("0001", 0) == 1
    assert bit_for_qubit("0001", 1) == 0
    assert bit_for_qubit("1000", 3) == 1
    assert bit_for_qubit("1000", 0) == 0


def test_circuit_has_two_registers_and_measurements():                         # T3
    qc = build_destructive_swap_circuit(N_QUBITS, THETA_A, THETA_B)
    assert qc.num_qubits == 2 * N_QUBITS
    assert qc.num_clbits == 2 * N_QUBITS


def test_bell_mapping_antisymmetric_singlet():                                 # T4
    # |Psi-> = (|01> - |10>) / sqrt(2)
    # Bell measurement: CNOT(0->1) then H(0)
    # CNOT -> (|01> - |11>) / sqrt(2) = (|0> - |1>)/sqrt(2) |1> = |->|1>
    # H(0)  -> |1>|1> = |11>
    psi_minus = Statevector(np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2))
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(0)
    out = psi_minus.evolve(qc)
    # Statevector index in Qiskit is little-endian: |11> is index 3
    assert abs(out.data[3]) == pytest.approx(1.0, abs=1e-9)


def test_bell_mapping_symmetric_triplets():                                    # T5
    # The three symmetric triplet states (|Phi+>, |Phi->, |Psi+>) must map to
    # outcomes other than 11 (|00>, |10>, |01>).
    phi_plus = Statevector(np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))
    phi_minus = Statevector(np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2))
    psi_plus = Statevector(np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2))

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(0)

    for state in [phi_plus, phi_minus, psi_plus]:
        out = state.evolve(qc)
        # Outcome 11 corresponds to basis index 3
        assert abs(out.data[3]) == pytest.approx(0.0, abs=1e-9)


def test_estimate_from_counts_pure_symmetric_outcome():                        # T6
    # Outcome "0000": no pair has (1,1), so eigenvalue = (+1)*(+1) = +1
    p11, pair_est, est, se = estimate_from_counts({"0000": 100}, 2)
    assert p11 == [0.0, 0.0]
    assert pair_est == [1.0, 1.0]
    assert est == pytest.approx(1.0)
    assert se == pytest.approx(0.0)


def test_estimate_from_counts_one_pair_antisymmetric():                        # T7
    # Pair 0 has (q0=1, q2=1); Pair 1 has (q1=0, q3=0)
    # In 4-qubit little-endian string: q3 q2 q1 q0 -> '0 1 0 1' = '0101'
    p11, pair_est, est, se = estimate_from_counts({"0101": 100}, 2)
    assert p11 == [1.0, 0.0]
    assert pair_est == [-1.0, 1.0]
    assert est == pytest.approx(-1.0)
    assert se == pytest.approx(0.0)


def test_estimate_from_counts_both_pairs_antisymmetric():                       # T8
    # Pair 0 has (q0=1, q2=1); Pair 1 has (q1=1, q3=1)
    # In 4-qubit little-endian string: q3 q2 q1 q0 -> '1 1 1 1' = '1111'
    # Global SWAP eigenvalue is (-1) * (-1) = +1
    p11, pair_est, est, se = estimate_from_counts({"1111": 100}, 2)
    assert p11 == [1.0, 1.0]
    assert pair_est == [-1.0, -1.0]
    assert est == pytest.approx(1.0)
    assert se == pytest.approx(0.0)


def test_one_call_interface_dict_keys_and_types():                             # T9
    r = estimate_overlap_destructive_swap(shots=500, seed=123, sampler="exact")
    expected_keys = {"method", "n", "shots", "estimate", "stderr", "theoretical", "abs_error"}
    assert set(r.keys()) == expected_keys
    assert r["method"] == "destructive_swap"
    assert isinstance(r["n"], int)
    assert isinstance(r["shots"], int)
    assert isinstance(r["estimate"], float)
    assert isinstance(r["stderr"], float)
    assert isinstance(r["theoretical"], float)
    assert isinstance(r["abs_error"], float)


def test_alias_interface():                                                    # T10
    r1 = estimate_overlap_destructive_swap(shots=500, seed=42, sampler="exact")
    r2 = estimate_overlap_swap(shots=500, seed=42, sampler="exact")
    assert r1 == r2


def test_reproducibility_with_seed_exact():                                    # T11
    r1 = estimate_overlap_destructive_swap(shots=1000, seed=999, sampler="exact")
    r2 = estimate_overlap_destructive_swap(shots=1000, seed=999, sampler="exact")
    assert r1["estimate"] == r2["estimate"]
    assert r1["stderr"] == r2["stderr"]


def test_reproducibility_with_seed_aer():                                      # T12
    r1 = estimate_overlap_destructive_swap(shots=1000, seed=888, sampler="aer")
    r2 = estimate_overlap_destructive_swap(shots=1000, seed=888, sampler="aer")
    assert r1["estimate"] == r2["estimate"]
    assert r1["stderr"] == r2["stderr"]


def test_exact_sampler_statistical_accuracy():                                 # T13
    r = estimate_overlap_destructive_swap(shots=50000, seed=1, sampler="exact")
    assert abs(r["estimate"] - 0.25) < 0.015


def test_aer_sampler_statistical_accuracy():                                   # T14
    r = estimate_overlap_destructive_swap(shots=20000, seed=1, sampler="aer")
    assert abs(r["estimate"] - 0.25) < 0.02


def test_custom_angles_identical_states():                                     # T15
    # Overlap of identical states must be 1.0 (delta = 0)
    theta = np.pi / 4
    r = estimate_overlap_destructive_swap(n=2, theta_a=theta, theta_b=theta,
                                          shots=5000, seed=42, sampler="exact")
    assert r["theoretical"] == pytest.approx(1.0)
    assert r["estimate"] == pytest.approx(1.0, abs=1e-6)


def test_custom_angles_orthogonal_states():                                    # T16
    # Overlap of orthogonal single-qubit states (delta = pi)
    # cos(pi/2)^(2*1) = 0
    r = estimate_overlap_destructive_swap(n=1, theta_a=0.0, theta_b=np.pi,
                                          shots=5000, seed=42, sampler="exact")
    assert r["theoretical"] == pytest.approx(0.0, abs=1e-12)
    assert abs(r["estimate"]) < 0.05
