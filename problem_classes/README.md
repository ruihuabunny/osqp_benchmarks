# OSQP optimisation instance generators

All **7 families of random optimisation instance generators in this directory have been reproduced and validated**: instance generation, standard QP assembly, OSQP solving, CVXPY models and solution recovery, reproducibility with fixed seeds, and parameter updates have all passed validation, with **46/46 checks passing**.

This document summarises the reproduction report and usage instructions. **The next phase aims to integrate these generators into [`src/generators/qp`](../../../src/generators/qp) in the main repository**, providing a source of QP instances for the project.

## 1. Reproduction report

### 1.1 Scope and environment

The reproduction covered the random generators listed below. Each family was tested at 2 sizes with `seed=0,1,2`, giving 42 base instances; 4 types of parameter updates were also validated.

| Generator | Problem type | Validated sizes | Base checks |
| --- | --- | --- | --- |
| `RandomQPExample` | Random QP with inequality constraints | `n=10,30` | 6 |
| `EqQPExample` | Equality-constrained QP | `n=10,30` | 6 |
| `PortfolioExample` | Factor-model portfolio optimisation | `k=3,5`, with the number of assets defaulting to `100*k` | 6 |
| `LassoExample` | L1-regularised least squares | Number of features `n=5,10` | 6 |
| `HuberExample` | Huber robust regression | Number of features `n=5,10` | 6 |
| `SVMExample` | Soft-margin support vector machine | Number of features `n=5,10` | 6 |
| `ControlExample` | Constrained optimal control / MPC | `nx=4,10`, `nu=floor(nx/2)`, `T=10` | 6 |

The saved results were recorded at **2026-09-07 12:11:34 UTC**, using Linux/WSL2 x86_64 and the `.venv` at the repository root.

| Dependency | Version used for reproduction |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| CVXPY | 1.9.2 |
| OSQP | 1.1.3 |

See [requirements-generators.lock](requirements-generators.lock) for the complete dependency snapshot and [generator_validation_results.json](generator_validation_results.json) for the raw measurements. The process and supporting evidence are documented in more detail in [GENERATORS_REPRODUCTION.md](GENERATORS_REPRODUCTION.md) and its [Notebook](GENERATORS_REPRODUCTION.ipynb).

### 1.2 Issues found and completed fixes

Before the changes, representative instances from all 7 families could be generated and solved, but some CVXPY expressions produced deprecation warnings, and Lasso parameter updates left the two models out of sync. Measurements from before the changes are saved in [generator_validation_baseline.json](generator_validation_baseline.json).

| Issue | Fix and validation result |
| --- | --- |
| CVXPY used `*` for matrix/vector multiplication | 6 generators now use `@`, with the corresponding summations changed to `cvxpy.sum()`; no warnings were captured in any of the 46 validation cases |
| Control called `cvxpy.vec()` without specifying the flattening order | Explicitly set `order='F'` to match the time ordering in the manually assembled QP |
| After a lambda update, the Lasso CVXPY objective still referenced the old constant | The objective now references a nonnegative `cvxpy.Parameter`; updates are synchronised across the QP and CVXPY models, and negative lambda values are rejected before modifying the QP |

For example, after doubling lambda for `LassoExample(10, seed=1)`, the direct QP and CVXPY objective values differed by approximately **8.23808** before the fix; after the fix, both were approximately **1034.2135226400615**. Validation also confirmed that the updated QP matched a QP reassembled from the current attributes.

These changes preserved the generator constructor interfaces, original problem structures, random generation procedures, sparse QP outputs, solution recovery, and parameter update methods. The existing reproduction report records that the initial QP data for all 42/42 base instances remained identical before and after the fixes.

### 1.3 Validation method and results

The standalone entry point is [validate_generators.py](validate_generators.py). Each base configuration undergoes the following checks:

1. Construct the instance and check the dimensions of `P,q,A,l,u,n,m`, sparse matrices, valid numerical values and bounds, Hessian symmetry, and CVXPY DCP compliance; where split bound fields are present, check the reconstructed constraints.
2. Call OSQP directly on the standard QP, then call OSQP through CVXPY; recover the CVXPY primal/dual solutions and substitute them into the original QP to check constraints, stationarity, complementarity residuals, and objective values.
3. Rebuild with the same seed and check that the data are identical; use `seed+100` to confirm that the data change. These comparison constructions are not counted as additional cases among the 42 solves.
4. For small Control instances, check that the dynamics spectral radius is less than 1 and that the initial state lies within the state bounds.

Solver settings are `eps_abs=eps_rel=1e-7`, `max_iter=100000`, `polishing=True`, `adaptive_rho_interval=50`, and `verbose=False`, with `warm_start=False` for CVXPY. The threshold for independently computed normalised residuals and objective differences is `5e-6`; see the validation script and detailed report for the definitions.

| Check category | Coverage | Result |
| --- | --- | --- |
| Base instances | 7 families × 2 sizes × 3 seeds | 42/42 passed |
| `lasso_lambda` | Double lambda; verify that the state remains unchanged after rejecting a negative value | Passed |
| `control_x0` | Scale the initial state by 0.5 | Passed |
| `portfolio_mu` | Update the return vector, with `k=3,n=60,seed=1` | Passed |
| `portfolio_F_D` | Multiply F by 0.9 and D by 1.1 while preserving the sparsity pattern | Passed |

All 46 checks returned `solved` from direct OSQP and `optimal` from CVXPY. The main errors in the saved results are shown below; maximum residuals include both solution paths.

| Metric | Maximum value |
| --- | --- |
| Primal constraint violation (absolute) | `2.887e-15` |
| Stationarity residual (absolute) | `5.800e-13` |
| Normalised primal residual | `9.021e-16` |
| Normalised stationarity residual | `9.111e-14` |
| Normalised complementarity residual | `2.605e-16` |
| Normalised objective difference between the two models | `2.073e-14` |

The recorded `run_suite()` took approximately **1.334 seconds**, including instance generation and validation but excluding Python startup and installation. This is the runtime of a small-scale correctness check, not a formal performance benchmark.

### 1.4 Scope of the conclusions

The completed reproduction covers the 7 random generator families and the tested configurations above. `maros_meszaros.py`, `qplib.py`, `suitesparse_lasso.py`, and `suitesparse_huber.py` are entry points for loading/converting external data and were not included in this validation; the corresponding `maros_meszaros_data/`, `qplib_data/`, and `suitesparse_matrix_collection/` datasets are not prerequisites for running these random generators.

Both solution paths use OSQP. This validation established consistency in model conversion and solution recovery; it did not reproduce all 1400 instances from the original paper, compare performance across multiple solvers, or validate scaling to larger instances.

## 2. Usage

### 2.1 Install dependencies and run validation

The following terminal commands are for Linux/WSL and should all be run from the **main repository root**. Skip the creation command if `.venv` already exists.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r third_parties/osqp_benchmarks/problem_classes/requirements-generators.lock
.venv/bin/python -m pip check
```

The generators depend on NumPy, SciPy, and CVXPY; direct solving uses OSQP. The lock file also includes the dependencies needed for the Notebooks. Once installed, generating and validating instances from the 7 random families requires no external data downloads.

```bash
# Full validation: 42 instances + 4 types of parameter updates; prints 46/46 passed on success
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --output /tmp/osqp-generators-rerun.json

# Quick validation: the first size for each family with seed=1, plus parameter updates; 11 checks in total
.venv/bin/python third_parties/osqp_benchmarks/problem_classes/validate_generators.py --quick
```

The script returns a nonzero exit code if validation fails. `--output` is optional; the path above writes the rerun results to `/tmp`, preserving the original reproduction records in this directory.

### 2.2 Constructor parameters and instance sizes

Instances are **generated when the class constructor is called**, which also creates `qp_problem` and `cvxpy_problem`. Constructors do not solve the problem or save files; running files directly, such as `python control.py`, only loads the class definitions.

| File and constructor | Parameter meanings | Final QP size `(n_QP, m_QP)` |
| --- | --- | --- |
| [random_qp.py](random_qp.py): `RandomQPExample(n, seed=1)` | `n` is the original number of variables | `(n, 10*n)` |
| [eq_qp.py](eq_qp.py): `EqQPExample(n, seed=1)` | `n` is the original number of variables | `(n, floor(n/2))` |
| [portfolio.py](portfolio.py): `PortfolioExample(k, seed=1, n=None)` | `k` is the number of factors; the number of assets `n` defaults to `100*k` | `(n+k, n+k+1)` |
| [lasso.py](lasso.py): `LassoExample(n, seed=1)` | `n` is the number of features; the number of samples is `100*n` | `(102*n, 102*n)` |
| [huber.py](huber.py): `HuberExample(n, seed=1)` | `n` is the number of features; the number of samples is `100*n` | `(301*n, 300*n)` |
| [svm.py](svm.py): `SVMExample(n, seed=1)` | `n` is the number of features; the number of samples is `100*n` | `(101*n, 200*n)` |
| [control.py](control.py): `ControlExample(n, seed=1)` | `nx=n`, `nu=floor(n/2)`, prediction horizon `T=10` | `((T+1)*nx+T*nu, 2*(T+1)*nx+T*nu)` |

Use explicit positive integer sizes, starting at `n>=2` for Eq QP and Control. To specify a custom number of assets for Portfolio, use `PortfolioExample(3, seed=1, n=60)`; its second positional argument is the seed. Control currently does not support specifying `nx/nu/T` independently through constructor parameters.

### 2.3 Generate instances and access the standard QP

Run the following Python code blocks in order, with the main repository root as the working directory and `.venv/bin/python` as the interpreter. In a Notebook, select the kernel from the same `.venv`.

```python
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "third_parties/osqp_benchmarks"))

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import osqp

from problem_classes.random_qp import RandomQPExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.lasso import LassoExample
from problem_classes.huber import HuberExample
from problem_classes.svm import SVMExample
from problem_classes.control import ControlExample

instances = {
    "random_qp": RandomQPExample(10, seed=1),
    "eq_qp": EqQPExample(10, seed=1),
    "portfolio": PortfolioExample(5, seed=1),
    "lasso": LassoExample(10, seed=1),
    "huber": HuberExample(10, seed=1),
    "svm": SVMExample(10, seed=1),
    "control": ControlExample(10, seed=1),
}
for name, example in instances.items():
    qp = example.qp_problem
    print(name, "n_QP=", qp["n"], "m_QP=", qp["m"],
          "nnz(P)=", qp["P"].nnz, "nnz(A)=", qp["A"].nnz)
```

All generators output a standard convex QP:

$$
\min_z\; \frac{1}{2}z^\mathsf{T}Pz + q^\mathsf{T}z,
\qquad l \leq Az \leq u.
$$

| `qp_problem` field | Meaning |
| --- | --- |
| `P` | SciPy sparse matrix of shape `(n_QP, n_QP)`, containing the full symmetric Hessian |
| `q` | One-dimensional NumPy array of shape `(n_QP,)` |
| `A` | SciPy sparse QP constraint matrix of shape `(m_QP, n_QP)` |
| `l`, `u` | One-dimensional lower and upper bound arrays of shape `(m_QP,)`, which may contain valid infinite bounds |
| `n`, `m` | Number of variables and constraint rows in the final QP, usually different from the constructor parameters |

Control, Portfolio, Huber, and SVM also provide `A_nobounds`, `l_nobounds`, `u_nobounds`, `bounds_idx`, `lx`, and `ux` to separate general constraints from variable bounds. Direct solving requires only `P,q,A,l,u`; do not unpack the entire dictionary into `OSQP.setup()`.

**For Control, `example.A` is the state-transition matrix, whereas `example.qp_problem['A']` is the QP constraint matrix.** Control QP variables are ordered as `[vec_F(x), vec_F(u)]`: the state and control trajectories are each flattened column by column in time order.

### 2.4 Solve and cross-check with CVXPY

```python
example = instances["lasso"]
qp = example.qp_problem
SETTINGS = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=100000,
                polishing=True, adaptive_rho_interval=50, verbose=False)

solver = osqp.OSQP()
solver.setup(**{key: qp[key] for key in ("P", "q", "A", "l", "u")}, **SETTINGS)
result = solver.solve(raise_error=True)
assert result.info.status == "solved", result.info.status
print("OSQP:", result.info.status, "objective:", result.info.obj_val)

example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
assert example.cvxpy_problem.status == cp.OPTIMAL
x_cvx, y_cvx = example.revert_cvxpy_solution()
assert x_cvx.shape == (qp["n"],) and y_cvx.shape == (qp["m"],)
np.testing.assert_allclose(result.info.obj_val, example.cvxpy_problem.value,
                           rtol=1e-6, atol=1e-6)
print("CVXPY:", example.cvxpy_problem.status, "objective:", example.cvxpy_problem.value)
```

The objective value for this Lasso instance is approximately `1025.9754409815434`. After successfully solving the CVXPY model, call `revert_cvxpy_solution()` to obtain primal and dual solutions in the ordering of the manually assembled QP. For complete residual checks, run the validation commands in Section 2.1.

### 2.5 Save and load instances

The following example saves the Lasso QP from the previous step: `P/A` use sparse NPZ files, while the vectors, dimensions, and seed use a NumPy NPZ file. Output is written to `/tmp/osqp-generator-example/lasso_n10_seed1`; rerunning the example overwrites files with the same names. Choose a different output directory for long-term storage.

```python
output_dir = Path("/tmp/osqp-generator-example/lasso_n10_seed1")
output_dir.mkdir(parents=True, exist_ok=True)
sp.save_npz(output_dir / "P.npz", qp["P"].tocsc())
sp.save_npz(output_dir / "A.npz", qp["A"].tocsc())
np.savez_compressed(output_dir / "vectors.npz", q=qp["q"], l=qp["l"], u=qp["u"],
                    n=qp["n"], m=qp["m"], seed=1)

with np.load(output_dir / "vectors.npz", allow_pickle=False) as arrays:
    loaded = {key: arrays[key].copy() for key in ("q", "l", "u")}
    loaded.update(n=int(arrays["n"].item()), m=int(arrays["m"].item()))
loaded.update(P=sp.load_npz(output_dir / "P.npz"), A=sp.load_npz(output_dir / "A.npz"))

loaded_solver = osqp.OSQP()
loaded_solver.setup(**{key: loaded[key] for key in ("P", "q", "A", "l", "u")}, **SETTINGS)
loaded_result = loaded_solver.solve(raise_error=True)
assert loaded_result.info.status == "solved"
np.testing.assert_allclose(loaded_result.info.obj_val, result.info.obj_val,
                           rtol=1e-6, atol=1e-6)
print("Saved, reloaded, and solved:", output_dir)
```

This export saves standard QP data, excluding CVXPY objects, domain attributes, and split bound fields; the five standard fields in the loaded data are sufficient to solve the problem again.

### 2.6 Parameter updates

The generators' update methods synchronise their internal QP/CVXPY models; OSQP objects already created by the user must be updated separately.

| Method | Parameter meanings and requirements | Action for an existing OSQP object |
| --- | --- | --- |
| `lasso.update_lambda(lambda_new)` | Nonnegative regularisation coefficient | `solver.update(q=lasso.qp_problem['q'])` |
| `control.update_x0(x0_new)` | Initial state of length `nx`; consider bounds and dynamics feasibility | `solver.update(l=control.qp_problem['l'], u=control.qp_problem['u'])` |
| `portfolio.update_parameters(mu)` | Return vector whose length equals the number of assets | `solver.update(q=portfolio.qp_problem['q'])` |
| `portfolio.update_parameters(mu, F=..., D=...)` | F/D must retain their original dimensions and CSC sparsity patterns, and D must preserve a convex quadratic cost | Create a new OSQP object and call `setup()` with the updated QP |

Continuing the previous Lasso example:

```python
example.update_lambda(2.0 * example.lambda_param)
solver.update(q=example.qp_problem["q"])
updated = solver.solve(raise_error=True)
assert updated.info.status == "solved"
example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
assert example.cvxpy_problem.status == cp.OPTIMAL
np.testing.assert_allclose(updated.info.obj_val, example.cvxpy_problem.value,
                           rtol=1e-6, atol=1e-6)
print("Objective after doubling lambda:", updated.info.obj_val)
```

The updated objective value is approximately `1034.2135226400615`. More examples and instructions for launching the Notebook are available in [GENERATORS_USAGE.md](GENERATORS_USAGE.md) and the executable [GENERATORS_USAGE.ipynb](GENERATORS_USAGE.ipynb).

### 2.7 Random seeds and scaling

The current generators use `np.random.seed(seed)`, which changes NumPy's global random state. A fixed dependency environment, constructor parameters, and seed reproduce the same instance; bitwise equality is not guaranteed across arbitrary dependency versions, and these constructors should not be called concurrently in threads because they depend on the global RNG.

Before increasing instance sizes, assess memory usage and generation cost based on the final QP dimensions. Lasso/Huber/SVM fix the number of samples at 100 times the number of features; the number of nonzeros in matrices with fixed density grows with their dimensions, and the Gram matrices in Random/Eq QP may also introduce fill-in. Control retains dense eigendecomposition and Riccati solving, so sparse QP assembly alone does not remove these bottlenecks. Feasibility and numerical behaviour at larger sizes and with other seeds require further validation.

## 3. Next phase: explore integration into `src/generators/qp`

**The next phase will explore integrating the 7 reproduced generator families in this directory into [`src/generators/qp`](../../../src/generators/qp) in the main repository, allowing the project to obtain standard QP instances from these problem families through its own generation entry points.** The target directory is currently empty, and the integration interface has yet to be implemented.

The suggested sequence is:

1. Review the main project's QP instance input/output requirements and define how to pass the problem family, size parameters, and seed, using the existing `P,q,A,l,u,n,m` fields as the data foundation.
2. Start with one generator and establish the complete workflow from invocation through `src/generators/qp` to generation and downstream use, then gradually integrate the other 6 families. Prioritise reusing the existing generation logic while preserving each family's mathematical structure and parameter meanings.
3. Preserve sparse storage for `P/A`, constraint and variable ordering, and necessary bound information. Record instance provenance using ordinary problem-family names, parameters, and seeds, and document dependencies and imports.
4. Under the same environment, parameters, and seed, compare QP data, objective values, and residuals before and after integration. Reuse the existing 46 validation checks and verify the workflow from the main project's entry point to downstream use.
5. Once basic integration passes validation, assess batch generation and size requirements, then use measured results to address limitations such as the global RNG, fixed dimension ratios, and Control's dense computations.

The acceptance target for this phase is for the main project to select any of the 7 problem families, specify the existing size parameters and seed, obtain a sparse QP that follows project conventions, and pass checks for consistency with the current reproduction baseline and for downstream use.
