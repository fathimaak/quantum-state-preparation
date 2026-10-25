# State Preparation Handoff (Fathima -> Sankhana, Haneen, Donita)

## 1. Single source of truth
Import everything from `states.py`. Do not retype the angles.

    from states import N_QUBITS, THETA_A, THETA_B, prepare_state, prepare_joint, theoretical_overlap

## 2. State definitions (Phase 1, ideal simulation)
- n = 2 qubits per register, pure product states with real amplitudes.
- State A = (Ry(pi/3)|0>)^(x2)      -> theta_A = pi/3  (1.047198 rad)
- State B = (Ry(5pi/6)|0>)^(x2)     -> theta_B = 5pi/6 (2.617994 rad), delta = pi/2
- Per qubit: Ry(t)|0> = cos(t/2)|0> + sin(t/2)|1>

## 3. Overlap definition and derivation
We estimate Tr(rho sigma) = |<A|B>|^2 (squared inner product). This is NOT the
amplitude |<A|B> | and NOT a general mixed-state fidelity.
- One qubit: <A|B> = cos(t_A/2)cos(t_B/2) + sin(t_A/2)sin(t_B/2) = cos(delta/2)
- So one-qubit overlap = cos^2(delta/2) = cos^2(pi/4) = 0.5
- Product states multiply across qubits: overlap = cos^(2n)(delta/2) = 0.5^2

**THEORETICAL OVERLAP = 0.25**   (checked against Aer: 0.250000)

## 4. Circuits
- `prepare_state(n, theta)`: n-qubit circuit, one Ry(theta) per qubit, no measurement.
- `prepare_joint()`: 4-qubit circuit, State A on qubits 0-1, State B on qubits 2-3.
  Teammates add their own measurement layer after it.

## 5. Conventions (please follow exactly)
- Register A = qubits 0..n-1, register B = qubits n..2n-1.
- Pair i is (A_i, B_i) = (qubit i, qubit n+i): pairs are (0,2) and (1,3).
- Qiskit bitstrings are little-endian: in counts keys, qubit 0 is the RIGHTMOST bit.
  With measure_all() on 4 qubits the key is q3 q2 q1 q0.
- Destructive SWAP (Bell measurement): CNOT(control A_i, target B_i), H on A_i, measure
  both; outcome A_i=1,B_i=1 means the antisymmetric state; <S_i> = 1 - 2 P(11).
- Check for Sankhana: per-pair <S_i> should be about 0.5, so P(11) = 0.25 per pair, and
  the product over both pairs is about 0.25. (Verified in ideal Aer: 0.4988, 0.5023, product 0.2532.)
- Shadows (Haneen): use the same states; per-qubit snapshot 3 U^dag|b><b|U - I; overlap
  estimate = cross-correlation of A and B snapshots (Sec. III of the paper).
  Note: the double sum over snapshots (j,k) is fine for A vs B. If a state is ever compared
  with itself (purity), the j = k terms must be excluded.
- Aer simulation is all-to-all connected, so routing noise will not appear in Phase 1.

## 6. Reference numbers for sanity checks (ideal)
- A amplitudes, basis order |q1 q0>: 00,01,10,11 = [0.75, 0.433013, 0.433013, 0.25]
- B amplitudes, same order:                        [0.066987, 0.25, 0.25, 0.933013]
- P(qubit = 1): qubits 0,1 (State A) = 0.25 ; qubits 2,3 (State B) = 0.933
- Compute-uncompute check, P(00) with 100000 shots: about 0.25 (+/- 0.004)

## 7. Donita
Error = |estimate - 0.25| for each method. Reference value: 0.25.