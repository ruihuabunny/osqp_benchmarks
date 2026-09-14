"""Numerical checks for the parameterized Eq QP and Lasso generators."""

import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from test_generators import SETTINGS, solve_and_check
from problem_classes.eq_qp import EqQPExample
from problem_classes.lasso import LassoExample


class GeneratorParameterTests(unittest.TestCase):
    def test_eq_qp_dimensions_and_density(self):
        default = EqQPExample(12, seed=1)
        self.assertEqual(default.qp_problem['m'], 6)
        for density in (0.2, np.float32(0.5), 1.0):
            with self.subTest(density=density):
                example = EqQPExample(12, m=np.int64(9),
                                      lower_density=density, seed=2)
                self.assertEqual(example.A.shape, (9, 12))
                self.assertEqual(example.A.nnz, round(9 * 12 * density))
                solve_and_check(example)

    def test_eq_qp_rank_and_spectral_bounds(self):
        cases = (
            {},
            dict(r=5),
            dict(r=5, max_spectrum=0.5),
            dict(r=5, max_cond_num=10.0),
            dict(r=5, max_spectrum=0.5, max_cond_num=10.0),
            dict(r=5, max_spectrum=2.0, max_cond_num=1.0),
            dict(r=0, max_spectrum=0.0),
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                example = EqQPExample(12, m=4, lower_density=0.3,
                                      seed=3, **kwargs)
                eigenvalues = np.linalg.eigvalsh(example.P.toarray())
                positive = eigenvalues[eigenvalues > 1e-9]
                self.assertGreaterEqual(eigenvalues[0], -1e-9)
                self.assertEqual(len(positive), example.r)
                if 'max_spectrum' in kwargs:
                    self.assertLessEqual(eigenvalues[-1],
                                         kwargs['max_spectrum'] + 1e-9)
                if 'max_cond_num' in kwargs:
                    self.assertLessEqual(positive[-1] / positive[0],
                                         kwargs['max_cond_num'] + 1e-9)
                solve_and_check(example)

    def test_lasso_rectangular_data_and_original_objective(self):
        for n, m, density in ((20, 8, 0.15), (12, 12, 1.0), (8, 35, 0.2)):
            with self.subTest(n=n, m=m, density=density):
                example = LassoExample(n, m=m, density=density,
                                       data_scale=2.0, lambda_ratio=0.2, seed=3)
                qp = example.qp_problem
                self.assertEqual(example.Ad.shape, (m, n))
                self.assertEqual(example.Ad.nnz, round(m * n * density))
                self.assertEqual((qp['n'], qp['m']), (m + 2*n, m + 2*n))
                result = solve_and_check(example)
                # Independently model the original objective without the
                # generator's explicit residual and epigraph variables.
                x = cp.Variable(n)
                original = cp.Problem(cp.Minimize(
                    cp.sum_squares(example.Ad @ x - example.bd)
                    + example.lambda_param * cp.norm1(x)))
                original.solve(solver=cp.OSQP, **SETTINGS)
                self.assertEqual(original.status, cp.OPTIMAL)
                self.assertAlmostEqual(result['osqp_objective'], original.value,
                                       delta=5e-6 * (1 + abs(original.value)))

    def test_lasso_data_scaling_preserves_solution(self):
        base = LassoExample(8, m=30, density=0.5, seed=4)
        solve_and_check(base)
        base_x = base.cvxpy_variables[0].value.copy()
        for scale in (0.2, 5.0):
            with self.subTest(scale=scale):
                scaled = LassoExample(8, m=30, density=0.5,
                                      data_scale=scale, seed=4)
                np.testing.assert_array_equal(scaled.Ad.indices, base.Ad.indices)
                np.testing.assert_array_equal(scaled.Ad.indptr, base.Ad.indptr)
                np.testing.assert_allclose(scaled.Ad.data, scale * base.Ad.data)
                np.testing.assert_allclose(scaled.bd, scale * base.bd)
                self.assertAlmostEqual(scaled.lambda_param,
                                       scale**2 * base.lambda_param)
                solve_and_check(scaled)
                np.testing.assert_allclose(scaled.cvxpy_variables[0].value,
                                           base_x, atol=1e-5, rtol=1e-5)

    def test_lasso_relative_regularization_threshold(self):
        for ratio in (0.99, 1.0, 2.0):
            with self.subTest(ratio=ratio):
                example = LassoExample(8, m=30, density=0.5,
                                       lambda_ratio=ratio, seed=5)
                result = solve_and_check(example)
                x = example.cvxpy_variables[0].value
                if ratio >= 1:
                    np.testing.assert_allclose(x, 0, atol=1e-6)
                    self.assertAlmostEqual(result['osqp_objective'],
                                           example.bd @ example.bd, places=5)
                else:
                    self.assertGreater(np.linalg.norm(x, np.inf), 1e-6)
                    self.assertLess(result['osqp_objective'],
                                    example.bd @ example.bd - 1e-6)

    def test_lasso_zero_regularization_and_update(self):
        example = LassoExample(8, m=30, density=1.0,
                               lambda_ratio=0.0, seed=6)
        solve_and_check(example)
        least_squares = np.linalg.lstsq(example.Ad.toarray(), example.bd,
                                       rcond=None)[0]
        np.testing.assert_allclose(example.cvxpy_variables[0].value,
                                   least_squares, atol=1e-6)
        example.update_lambda(1.5 * example.lambda_max)
        solve_and_check(example)
        np.testing.assert_allclose(example.cvxpy_variables[0].value, 0, atol=1e-6)

    def test_lasso_large_sparse_generation(self):
        example = LassoExample(10000, m=20000, density=0.0005, seed=7)
        qp = example.qp_problem
        self.assertEqual(example.Ad.nnz, 100000)
        self.assertEqual((qp['n'], qp['m']), (40000, 40000))
        self.assertTrue(sp.isspmatrix_csc(qp['P']))
        self.assertTrue(sp.isspmatrix_csc(qp['A']))
        self.assertEqual(qp['P'].nnz, 20000)
        self.assertEqual(qp['A'].nnz, 160000)

    def test_reproducibility_without_global_rng_changes(self):
        for cls in (EqQPExample, LassoExample):
            with self.subTest(generator=cls.__name__):
                before = np.random.get_state()
                first = cls(12, m=5, seed=8)
                after = np.random.get_state()
                self.assertEqual(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                self.assertEqual(before[2:], after[2:])
                same = cls(12, m=5, seed=8)
                different = cls(12, m=5, seed=9)
                for key in ('P', 'A'):
                    a, b = first.qp_problem[key], same.qp_problem[key]
                    np.testing.assert_array_equal(a.indptr, b.indptr)
                    np.testing.assert_array_equal(a.indices, b.indices)
                    np.testing.assert_array_equal(a.data, b.data)
                for key in ('q', 'l', 'u'):
                    np.testing.assert_array_equal(first.qp_problem[key],
                                                  same.qp_problem[key])
                self.assertGreater((first.qp_problem['A']
                                    - different.qp_problem['A']).nnz, 0)

    def test_lasso_legacy_call(self):
        positional = LassoExample(5, 9)
        keyword = LassoExample(5, seed=9)
        self.assertEqual(positional.m, 500)
        np.testing.assert_array_equal(positional.bd, keyword.bd)
        self.assertAlmostEqual(positional.lambda_param,
                               0.2 * np.linalg.norm(
                                   positional.Ad.T @ positional.bd, np.inf))

    def test_invalid_parameters(self):
        eq_cases = (
            dict(m=0), dict(m=13), dict(m=2.5),
            dict(lower_density=0), dict(lower_density=1.1),
            dict(lower_density=np.nan), dict(r=13),
            dict(max_spectrum=-1), dict(max_cond_num=0.5),
        )
        lasso_cases = (
            dict(m=0), dict(m=2.5), dict(density=0), dict(density=1.1),
            dict(density=np.nan), dict(data_scale=0), dict(data_scale=np.inf),
            dict(data_scale=np.nan), dict(lambda_ratio=-1),
            dict(lambda_ratio=np.inf), dict(lambda_ratio=np.nan),
        )
        for cls, cases in ((EqQPExample, eq_cases), (LassoExample, lasso_cases)):
            for kwargs in cases:
                with self.subTest(generator=cls.__name__, **kwargs):
                    with self.assertRaises(ValueError):
                        cls(12, **kwargs)


if __name__ == '__main__':
    unittest.main()
