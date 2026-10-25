"""test_shadows.py - run with:  python -m pytest -v"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from states import N_QUBITS, THETA_A, THETA_B, theoretical_overlap
from shadows import (add_basis_rotation, parse_memory, collect_snapshots_aer,
                     collect_snapshots_exact, estimate_overlap_from_snapshots,
                     estimate_overlap_shadows)


def test_reference_value_from_states_py():                                   # T1
    assert theoretical_overlap() == pytest.approx(0.25)


def test_bit_order_is_little_endian():                                       # T2
    assert parse_memory("0001") == [1, 0, 0, 0]          # qubit 0 is the RIGHTMOST character
    assert parse_memory("1000") == [0, 0, 0, 1]


def _eigenstate(axis, sign):
    """+/-1 eigenstate of X (0), Y (1), Z (2) as a statevector."""
    vecs = {(0, 1): [1, 1], (0, -1): [1, -1], (1, 1): [1, 1j], (1, -1): [1, -1j],
            (2, 1): [1, 0], (2, -1): [0, 1]}
    v = np.array(vecs[(axis, sign)], dtype=complex)
    return Statevector(v / np.linalg.norm(v))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [1, -1])
def test_basis_rotation_maps_eigenstate_to_computational_state(axis, sign):  # T3
    qc = QuantumCircuit(1)
    add_basis_rotation(qc, [axis])
    out = _eigenstate(axis, sign).evolve(qc)
    expected_bit = 0 if sign == 1 else 1                  # outcome 0 <-> +1, outcome 1 <-> -1
    assert abs(out.data[expected_bit]) == pytest.approx(1.0, abs=1e-9)


def test_snapshot_pair_factor_matches_projector_algebra():                   # T4
    """f = 1/2 + 9/2 * sA * sB * [same axis]  must equal 9|<a|b>|^2 - 4 computed from states."""
    for ax_a in range(3):
        for ax_b in range(3):
            for s_a in (1, -1):
                for s_b in (1, -1):
                    exact = 9 * abs(_eigenstate(ax_a, s_a).inner(_eigenstate(ax_b, s_b))) ** 2 - 4
                    formula = 0.5 + 4.5 * s_a * s_b * (ax_a == ax_b)
                    assert formula == pytest.approx(exact, abs=1e-9)


def test_estimator_is_unbiased():                                            # T5
    ests = [estimate_overlap_shadows(shots=300, seed=s, sampler="exact")["estimate"]
            for s in range(1500)]
    mean, sem = np.mean(ests), np.std(ests, ddof=1) / np.sqrt(len(ests))
    assert abs(mean - 0.25) < 4 * sem


def test_reported_standard_error_is_calibrated():                            # T6
    runs = [estimate_overlap_shadows(shots=500, seed=s, sampler="exact") for s in range(400)]
    emp = np.std([r["estimate"] for r in runs], ddof=1)
    rep = np.mean([r["stderr"] for r in runs])
    assert rep == pytest.approx(emp, rel=0.15)


@pytest.mark.parametrize("delta,expected", [(0.0, 1.0), (np.pi, 0.0)])
def test_extreme_overlaps(delta, expected):                                  # T7
    r = estimate_overlap_shadows(theta_b=THETA_A + delta, shots=6000, seed=11, sampler="exact")
    assert r["theoretical"] == pytest.approx(expected, abs=1e-9)
    assert abs(r["estimate"] - expected) < 5 * r["stderr"] + 0.02


def test_aer_pipeline_agrees_with_theory():                                  # T8
    r = estimate_overlap_shadows(shots=4000, seed=21, sampler="aer")
    assert abs(r["estimate"] - r["theoretical"]) < 5 * r["stderr"]


def test_aer_and_exact_samplers_agree_statistically():                       # T9
    a = estimate_overlap_shadows(shots=8000, seed=3, sampler="aer")
    b = estimate_overlap_shadows(shots=8000, seed=4, sampler="exact")
    assert abs(a["estimate"] - b["estimate"]) < 5 * np.hypot(a["stderr"], b["stderr"])


def test_same_seed_gives_same_result():                                      # T10
    a = estimate_overlap_shadows(shots=500, seed=99, sampler="aer")
    b = estimate_overlap_shadows(shots=500, seed=99, sampler="aer")
    assert a["estimate"] == b["estimate"]


def test_interface_keys_for_donita():                                        # T11
    r = estimate_overlap_shadows(shots=200, seed=1, sampler="exact")
    assert set(r) == {"method", "n", "shots", "estimate", "stderr", "theoretical", "abs_error"}
    assert r["method"] == "classical_shadows" and r["shots"] == 200
