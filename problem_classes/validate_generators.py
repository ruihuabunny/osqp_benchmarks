"""Reproduce the seven synthetic generators without loading external datasets."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import warnings

import cvxpy as cp
import numpy as np
import osqp
import scipy.sparse as sp

# Support both direct execution and import from the accompanying notebooks.
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from problem_classes.control import ControlExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.huber import HuberExample
from problem_classes.lasso import LassoExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.random_qp import RandomQPExample
from problem_classes.svm import SVMExample


CASES = (
    (RandomQPExample, (10, 30)),
    (EqQPExample, (10, 30)),
    (PortfolioExample, (3, 5)),
    (LassoExample, (5, 10)),
    (HuberExample, (5, 10)),
    (SVMExample, (5, 10)),
    (ControlExample, (4, 10)),
)
SETTINGS = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=100000,
                polishing=True, adaptive_rho_interval=50, verbose=False)
CHECK_TOL = 5e-6


def qp_fingerprint(qp):
    """Hash all QP fields, normalizing sparse storage without making it dense."""
    digest = hashlib.sha256()
    for key in sorted(qp):
        value = qp[key]
        digest.update(key.encode())
        if sp.issparse(value):
            value = value.tocsc(copy=True)
            value.sum_duplicates()
            value.eliminate_zeros()
            value.sort_indices()
            arrays = (np.asarray(value.shape, dtype='<i8'),
                      value.indptr.astype('<i8'), value.indices.astype('<i8'),
                      value.data.astype('<f8'))
        else:
            value = np.asarray(value)
            arrays = (np.asarray(value.shape, dtype='<i8'), value.astype('<f8'))
        for array in arrays:
            digest.update(array.tobytes())
    return digest.hexdigest()


def validate_qp(qp):
    n, m = qp['n'], qp['m']
    for key, shape in (('P', (n, n)), ('A', (m, n))):
        matrix = qp[key]
        assert sp.issparse(matrix), f'{key} must remain sparse'
        assert matrix.shape == shape, (key, matrix.shape, shape)
        assert np.isrealobj(matrix.data) and np.isfinite(matrix.data).all(), key
    for key, shape in (('q', (n,)), ('l', (m,)), ('u', (m,))):
        assert np.asarray(qp[key]).shape == shape, key
        assert not np.isnan(qp[key]).any(), key
    assert np.isfinite(qp['q']).all()
    assert not np.isposinf(qp['l']).any()
    assert not np.isneginf(qp['u']).any()
    assert np.all(qp['l'] <= qp['u'])
    skew = qp['P'] - qp['P'].T
    assert skew.nnz == 0 or np.max(np.abs(skew.data)) < 1e-10
    if 'A_nobounds' in qp:
        matrix = qp['A_nobounds']
        assert sp.issparse(matrix) and matrix.shape[1] == n
        assert qp['l_nobounds'].shape == qp['u_nobounds'].shape == (matrix.shape[0],)
        assert qp['lx'].shape == qp['ux'].shape == (n,)
        indices = qp['bounds_idx']
        assert np.all((0 <= indices) & (indices < n))
        assert len(np.unique(indices)) == len(indices)
        identity_rows = sp.eye(n, format='csc')[indices, :]
        reconstructed = dict(P=qp['P'], q=qp['q'],
                             A=sp.vstack([matrix, identity_rows], format='csc'),
                             l=np.concatenate([qp['l_nobounds'], qp['lx'][indices]]),
                             u=np.concatenate([qp['u_nobounds'], qp['ux'][indices]]))
        original = {key: qp[key] for key in reconstructed}
        assert qp_fingerprint(original) == qp_fingerprint(reconstructed)


def residuals(qp, x, y):
    """Independently check feasibility, stationarity, and complementarity."""
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == (qp['n'],) and y.shape == (qp['m'],)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    ax, px, aty = qp['A'] @ x, qp['P'] @ x, qp['A'].T @ y
    norm = lambda value: float(np.linalg.norm(value, np.inf))
    primal = max(0.0, float(np.max(qp['l'] - ax)), float(np.max(ax - qp['u'])))
    dual = norm(px + qp['q'] + aty)
    positive, negative = np.maximum(y, 0), np.maximum(-y, 0)
    # min(multiplier, slack) is zero exactly at complementary slackness.
    comp = max(norm(np.minimum(positive, np.abs(qp['u'] - ax))),
               norm(np.minimum(negative, np.abs(ax - qp['l']))))
    values = dict(primal_abs=primal, dual_abs=dual, complementarity_abs=comp,
                  primal_scaled=primal / (1 + norm(ax)),
                  dual_scaled=dual / (1 + max(norm(px), norm(qp['q']), norm(aty))),
                  complementarity_scaled=comp / (1 + max(norm(ax), norm(y))))
    for key in ('primal_scaled', 'dual_scaled', 'complementarity_scaled'):
        assert values[key] <= CHECK_TOL, (key, values[key])
    return values


def solve_and_check(example):
    qp = example.qp_problem
    validate_qp(qp)
    assert example.cvxpy_problem.is_dcp(), 'CVXPY model is not DCP'
    solver = osqp.OSQP()
    solver.setup(**{key: qp[key] for key in ('P', 'q', 'A', 'l', 'u')}, **SETTINGS)
    started = perf_counter()
    result = solver.solve(raise_error=True)
    direct_seconds = perf_counter() - started
    assert result.info.status == 'solved', result.info.status
    direct_residuals = residuals(qp, result.x, result.y)
    started = perf_counter()
    example.cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, **SETTINGS)
    cvxpy_seconds = perf_counter() - started
    assert example.cvxpy_problem.status == cp.OPTIMAL, example.cvxpy_problem.status
    x, y = example.revert_cvxpy_solution()
    cvxpy_residuals = residuals(qp, x, y)
    direct_objective = float(.5 * result.x @ (qp['P'] @ result.x) + qp['q'] @ result.x)
    reverted_objective = float(.5 * x @ (qp['P'] @ x) + qp['q'] @ x)
    cvxpy_objective = float(example.cvxpy_problem.value)
    scale = 1 + max(abs(direct_objective), abs(cvxpy_objective))
    objective_gap = abs(direct_objective - cvxpy_objective) / scale
    assert objective_gap <= CHECK_TOL, ('objective_gap', objective_gap)
    assert abs(reverted_objective - cvxpy_objective) / scale <= CHECK_TOL
    return dict(n_qp=qp['n'], m_qp=qp['m'], nnz_P=qp['P'].nnz,
                nnz_A=qp['A'].nnz, qp_sha256=qp_fingerprint(qp),
                osqp_status=result.info.status, cvxpy_status=example.cvxpy_problem.status,
                osqp_objective=direct_objective, cvxpy_objective=cvxpy_objective,
                objective_gap_scaled=objective_gap, osqp_iterations=result.info.iter,
                direct_solve_seconds=direct_seconds, cvxpy_solve_seconds=cvxpy_seconds,
                direct_residuals=direct_residuals, cvxpy_residuals=cvxpy_residuals)


def run_case(cls, size, seed):
    started = perf_counter()
    example = cls(size, seed=seed)
    generate_seconds = perf_counter() - started
    fingerprint = qp_fingerprint(example.qp_problem)
    assert fingerprint == qp_fingerprint(cls(size, seed=seed).qp_problem), 'Seed not reproducible'
    assert fingerprint != qp_fingerprint(cls(size, seed=seed + 100).qp_problem), 'Seed not effective'
    result = solve_and_check(example)
    if isinstance(example, ControlExample):
        # Only small validation sizes: do not use a dense eigensolve for huge MPCs.
        radius = float(np.max(np.abs(np.linalg.eigvals(example.A.toarray()))))
        assert radius < 1
        assert np.all(example.xmin <= example.x0) and np.all(example.x0 <= example.xmax)
        result['dynamics_spectral_radius'] = radius
    return dict(result, generate_seconds=generate_seconds,
                same_seed_equal=True, different_seed_different=True)


def run_update(name):
    if name == 'lasso_lambda':
        example = LassoExample(10, seed=1)
    elif name == 'control_x0':
        example = ControlExample(10, seed=1)
    else:
        example = PortfolioExample(3, seed=1, n=60)
    before = solve_and_check(example)
    if name == 'lasso_lambda':
        example.update_lambda(2 * example.lambda_param)
        fingerprint = qp_fingerprint(example.qp_problem)
        previous_lambda = example.lambda_param
        try:
            example.update_lambda(-1.0)
        except ValueError:
            pass
        else:
            raise AssertionError('Negative lambda was accepted')
        assert qp_fingerprint(example.qp_problem) == fingerprint
        assert example.lambda_param == example.cvxpy_param.value == previous_lambda
    elif name == 'control_x0':
        example.update_x0(.5 * example.x0)
    elif name == 'portfolio_mu':
        example.update_parameters(example.mu + np.linspace(0, .2, example.n))
    elif name == 'portfolio_F_D':
        example.update_parameters(example.mu.copy(), F=.9 * example.F, D=1.1 * example.D)
    else:
        raise ValueError(name)
    assert before['qp_sha256'] != qp_fingerprint(example.qp_problem)
    assert qp_fingerprint(example.qp_problem) == qp_fingerprint(example._generate_qp_problem())
    return dict(solve_and_check(example), before_objective=before['osqp_objective'],
                rebuilt_qp_equal=True, **({'negative_lambda_rejected_without_mutation': True}
                                         if name == 'lasso_lambda' else {}))


def capture_case(label, function, **metadata):
    result = dict(case=label, **metadata)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result.update(function())
            result['passed'] = True
        except Exception:
            result.update(passed=False, error=traceback.format_exc())
        result['warnings'] = sorted({item.category.__name__ + ': ' + str(item.message)
                                     for item in caught})
    print(f"{'PASS' if result['passed'] else 'FAIL'} {label}", flush=True)
    if not result['passed']:
        print(result['error'], flush=True)
    return result


def run_suite(quick=False):
    """Return measured results; the caller chooses whether to save them."""
    started = perf_counter()
    records = []
    for cls, sizes in CASES:
        for size in (sizes[:1] if quick else sizes):
            for seed in ((1,) if quick else (0, 1, 2)):
                label = f'{cls.__name__}(size={size}, seed={seed})'
                records.append(capture_case(label, lambda: run_case(cls, size, seed),
                                            kind='generator', class_name=cls.__name__,
                                            size=size, seed=seed))
    for name in ('lasso_lambda', 'control_x0', 'portfolio_mu', 'portfolio_F_D'):
        records.append(capture_case(name, lambda: run_update(name), kind='parameter_update'))
    source_files = [Path(__file__)] + [Path(__file__).with_name(cls.__module__.split('.')[-1] + '.py')
                                     for cls, _ in CASES]
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, executable=sys.executable, prefix=sys.prefix,
                platform=platform.platform(),
                versions={name: version(name) for name in
                          ('numpy', 'scipy', 'cvxpy', 'osqp', 'nbformat', 'nbclient',
                           'ipykernel', 'jupyterlab')},
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_files},
                settings=SETTINGS, check_tolerance=CHECK_TOL, quick=quick,
                elapsed_seconds=perf_counter() - started, results=records,
                passed=sum(record['passed'] for record in records), total=len(records),
                all_passed=all(record['passed'] for record in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='One size and seed per class, plus updates')
    parser.add_argument('--output', type=Path, help='Optional measured-result JSON file')
    args = parser.parse_args()
    result = run_suite(quick=args.quick)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f"{result['passed']}/{result['total']} passed in {result['elapsed_seconds']:.3f}s")
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
