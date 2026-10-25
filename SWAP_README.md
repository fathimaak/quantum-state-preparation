# Destructive SWAP Test overlap estimator (owner: Sankhana)

Estimates the overlap `Tr(rho sigma) = |<A|B>|^2` of State A and State B from `states.py`
using transversal Bell-basis measurements (CNOT + Hadamard) on corresponding qubit pairs.

## Files
| File | Purpose |
|---|---|
| `destructive_swap.py` | The method: transversal Bell circuit, Aer / exact counts, pair & global overlap estimators, standard error, one-call interface |
| `destructive_swap_demo.py` | One readable run (2000 shots) with pair probabilities and standard error |
| `destructive_swap_experiment.py` | Accuracy-versus-shots sweep, writes `results/*.csv` |
| `plot_destructive_swap.py` | Quick error-versus-shots plot, saves `results/*.png` |
| `test_destructive_swap.py` | 16 pytest checks (run `python -m pytest -v`) |

## Run
```bash
python -m pytest -v
python destructive_swap_demo.py
python destructive_swap_experiment.py --trials 20
python plot_destructive_swap.py
```

## Interface for Donita / Haneen
```python
from destructive_swap import estimate_overlap_destructive_swap  # or estimate_overlap_swap

r = estimate_overlap_destructive_swap(shots=1000, seed=1)
# r = {method, n, shots, estimate, stderr, theoretical, abs_error}
```

`shots` is the TOTAL number of measurement shots. The sweep uses shots in `[100, 200, 500, 1000, 2000, 5000, 10000]`
and 20 trials per budget with seeds `1000 * shots + trial`. This matches Haneen's classical shadows budgets for equal-cost comparison.

## CSV schema

### Detailed results (`results/destructive_swap_results.csv`)
`method,n,shots,trial,seed,estimate,stderr,theoretical,abs_error`

- `method`: `"destructive_swap"`
- `n`: qubits per register (2)
- `shots`: shot budget
- `trial`: trial index (0 to trials-1)
- `seed`: reproducible random seed (`1000 * shots + trial`)
- `estimate`: estimated overlap $\text{Tr}(\rho \sigma)$
- `stderr`: reported standard error of the mean
- `theoretical`: closed-form overlap reference ($0.25$)
- `abs_error`: $|\text{estimate} - \text{theoretical}|$

### Summary metrics (`results/destructive_swap_summary.csv`)
`shots,mean_estimate,empirical_std,mean_abs_error,mean_reported_se,rmse`

- `mean_estimate`: mean of estimate across trials
- `empirical_std`: sample standard deviation of estimate across trials (ddof=1)
- `mean_abs_error`: mean absolute error across trials
- `mean_reported_se`: mean reported standard error
- `rmse`: root mean square error $\sqrt{\mathbb{E}[(\text{estimate} - \text{theoretical})^2]}$

## Conventions (same as HANDOFF.md)
* Register A = qubits 0..n-1, register B = qubits n..2n-1. Pairs are (0, 2) and (1, 3).
* Qiskit strings are little-endian: qubit 0 is the rightmost character.
* Pair $i$: CNOT(control $A_i$, target $B_i$), $H$ on $A_i$. Outcome $A_i=1, B_i=1$ is the antisymmetric state. $\langle S_i \rangle = 1 - 2 P(11)$.
* For product states, global SWAP expectation is the product of pair eigenvalues evaluated per shot.
* Ideal Aer simulation only (no noise, all-to-all connectivity) in Phase 1.
