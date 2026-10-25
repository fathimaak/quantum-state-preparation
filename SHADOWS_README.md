# Classical Shadows overlap estimator (owner: Haneen)

Estimates the overlap `Tr(rho sigma) = |<A|B>|^2` of State A and State B from `states.py`
using only random single-qubit measurements (no gate connects register A to register B).

## Files
| File | Purpose |
|---|---|
| `shadows.py` | The method: random bases, Aer snapshots, overlap estimator, standard error, one-call interface |
| `shadows_demo.py` | One readable run (2000 shots) |
| `shadows_experiment.py` | Accuracy-versus-shots sweep, writes `results/*.csv` |
| `plot_shadows.py` | Quick error-versus-shots plot |
| `test_shadows.py` | 17 pytest checks (run `python -m pytest -v`) |

## Run
    pip install -r requirements.txt
    python -m pytest -v
    python shadows_demo.py
    python shadows_experiment.py --trials 20
    python plot_shadows.py

## Interface for Donita / Sankhana
    from shadows import estimate_overlap_shadows
    r = estimate_overlap_shadows(shots=1000, seed=1)
    # r = {method, n, shots, estimate, stderr, theoretical, abs_error}

`shots` is the TOTAL number of measurement shots. The sweep uses shots in 100, 200, 500, 1000, 2000, 5000, 10000
and 20 trials per budget with seeds `1000 * shots + trial`. Please use the same budgets for the SWAP method so the
comparison is at equal cost.

## CSV schema (results/shadows_results.csv)
`method,n,shots,trial,seed,estimate,stderr,theoretical,abs_error`

## Conventions (same as HANDOFF.md)
* Register A = qubits 0..n-1, register B = qubits n..2n-1. Pairs are (0,2) and (1,3).
* Qiskit strings are little-endian: qubit 0 is the rightmost character.
* The estimate is unbiased but NOT clipped to [0, 1]; small negative values or values slightly above 1 are normal.
* Ideal Aer simulation only (no noise, all-to-all connectivity) in Phase 1.
