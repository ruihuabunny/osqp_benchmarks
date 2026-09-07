# AGENTS.md

## Scope

This file applies to `problem_classes/`.

## Current Objective

The current goal is to **refactor and scale up the instance-generation logic in the existing problem classes**, so that they can generate substantially larger optimisation instances while preserving the mathematical structure of each original problem family.

Do not modify the OSQP solver itself. Focus on instance generation.

The first important target is `control.py`.

---

## `control.py`

Before modifying `control.py`, preserve the meaning of the existing variables.

The `ControlExample` generator currently uses:

```python
self.nx      # number of states
self.nu      # number of control inputs

self.A       # state-transition / dynamics matrix
self.B       # control-input matrix

self.Q       # state cost matrix
self.R       # input cost matrix
self.QN      # terminal state cost matrix

self.xmin
self.xmax    # state bounds

self.umin
self.umax    # control-input bounds

self.x0      # initial state
self.T       # MPC horizon
```

The optimisation variables in the CVXPY formulation are:

```python
x  # state trajectory, shape (nx, T + 1)
u  # control trajectory, shape (nu, T)
```

### Important naming distinction

Do not confuse:

```python
self.A
```

with the local variable:

```python
A
```

inside `_generate_qp_problem()`.

`self.A` is the **state dynamics matrix** in

```python
x[:, i + 1] = self.A @ x[:, i] + self.B @ u[:, i]
```

whereas the local `A` assembled inside `_generate_qp_problem()` is the final **QP constraint matrix**.

Preserve this distinction when refactoring.

---

## Current Control Generator Structure

The generator currently performs roughly:

```text
n
↓
nx = n
nu = n / 2
↓
generate dynamics matrices self.A, self.B
↓
generate cost matrices self.Q, self.R, self.QN
↓
generate state/input bounds
↓
generate initial state self.x0
↓
set horizon self.T
↓
assemble sparse MPC QP
```

The final QP is stored in:

```python
self.qp_problem
```

and currently contains:

```python
P
q
A
l
u
m
n
A_nobounds
l_nobounds
u_nobounds
bounds_idx
lx
ux
```

Preserve this interface where practical.

---

## Large-Scale Refactoring Goal

The main goal is to make the existing generators capable of producing **large-scale but mathematically meaningful instances**.

Do not simply multiply dimensions without considering memory or computational complexity.

For `control.py`, scaling should primarily consider:

```python
self.nx
self.nu
self.T
```

and the structure/sparsity of:

```python
self.A
self.B
self.Q
self.R
self.QN
```

The resulting MPC/QP structure must remain a control problem.

Do not replace it with an unrelated random QP.

---

## Sparse Computation

Large matrices should remain sparse whenever possible.

Prefer SciPy sparse constructions such as:

```python
spa.eye
spa.diags
spa.kron
spa.block_diag
spa.vstack
spa.hstack
spa.csc_matrix
```

The existing `_generate_qp_problem()` already relies heavily on sparse Kronecker/block constructions. Preserve this structure.

Avoid unnecessarily converting large matrices to dense arrays.

---

## Known Scaling Bottlenecks in `control.py`

Pay particular attention to the existing generation of `self.A`:

```python
lambda_values, V = np.linalg.eig(self.A.todense())
```

and terminal-cost generation:

```python
QN = sla.solve_discrete_are(
    self.A.todense(),
    self.B.todense(),
    self.Q.todense(),
    self.R.todense(),
)
```

These operations require dense matrices and may become major bottlenecks when `self.nx` becomes large.

When refactoring for large-scale generation, investigate constructions that preserve the intended properties without requiring dense `O(nx^2)` storage or routine `O(nx^3)` linear algebra.

Do not change the mathematical meaning of the control problem merely to remove these operations.

---

## Dynamics Stability

The current generator modifies `self.A` so that its eigenvalues have magnitude below approximately 1.

Any replacement large-scale generation method must preserve the purpose of this step:

> generate a numerically reasonable discrete-time dynamics matrix with controlled stability.

Prefer generating `self.A` with the desired spectral/stability properties by construction if that avoids a full dense eigendecomposition.

---

## Cost Matrices

Preserve the roles of:

```python
self.Q
self.R
self.QN
```

as state, input, and terminal costs.

They must remain suitable for constructing a convex control QP.

Do not generate arbitrary indefinite cost matrices.

For large-scale generation, prefer sparse or structured cost matrices where appropriate.

---

## Bounds and Feasibility

Preserve the meanings of:

```python
self.xmin
self.xmax
self.umin
self.umax
self.x0
```

The initial state should remain compatible with the state bounds.

The generated control problem should be valid and, when intended by the original generator, feasible.

---

## Reproducibility

The generator must remain reproducible from `seed`.

For substantially refactored random generation, prefer a local RNG:

```python
rng = np.random.default_rng(seed)
```

rather than modifying NumPy's global random state.

The same parameters and seed should reproduce the same instance.

---

## Backward Compatibility

Where practical, preserve:

```python
ControlExample(n, seed=...)
```

and existing public attributes and methods, including:

```python
self.qp_problem
self.cvxpy_problem
self.cvxpy_variables
self.cvxpy_param

_generate_qp_problem()
_generate_cvxpy_problem()
revert_cvxpy_solution()
update_x0()
```

Do not change downstream benchmark-facing interfaces unless required for large-scale generation.

---

## Other Problem Classes

After understanding each file, apply the same principles to other generators under `problem_classes/`.

Do not assume that variables or structures used by `control.py` apply to another problem class.

Before modifying each generator:

1. read the complete source file;
2. identify its actual instance-generation variables;
3. understand the optimisation formulation;
4. identify scaling bottlenecks;
5. preserve the original problem structure;
6. then implement large-scale generation.

Do not invent generic parameter names or abstractions before inspecting the actual implementation.

---

## Non-Goals

Do not:

* modify OSQP solver algorithms;
* redesign convergence logic;
* broadly refactor benchmark infrastructure;
* rewrite unrelated code;
* implement downstream agentic-RL components here.

The current task is specifically:

> **Refactor the existing `problem_classes` instance generators to support meaningful large-scale optimisation instances while preserving their original mathematical structure and interfaces.**

---

## Priority

When making trade-offs, prioritize:

1. mathematical correctness;
2. large-scale scalability;
3. sparse-memory efficiency;
4. numerical stability;
5. preservation of the original problem structure;
6. reproducibility;
7. backward compatibility.
