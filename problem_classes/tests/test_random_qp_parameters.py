"""Small numerical checks for the block-structured random QP generator."""

import unittest

import numpy as np
import scipy.sparse as sp

from test_generators import solve_and_check, validate_qp
from problem_classes.random_qp import RandomQPExample


def make_random_qp(**overrides):
    parameters = dict(n=9, m=13, P_block_sizes=[2, 4, 3], P_rank=9,
                      P_lambda_max=5.0, P_cond_num=20.0, A_density=0.4,
                      q_scale=1.0, slack_scale=1.0, seed=2)
    parameters.update(overrides)
    return RandomQPExample(**parameters)


class RandomQPParameterTests(unittest.TestCase):
    def test_spectrum_range_feasibility_and_statistics(self):
        cases = (
            {},
            dict(P_rank=4, m=5),
            dict(P_rank=1, P_block_sizes=[1] * 9),
            dict(P_cond_num=1.0, P_block_sizes=[9]),
            dict(P_rank=4, P_cond_num=1.0),
            dict(n=1, m=2, P_block_sizes=[1], P_rank=1),
        )
        for parameters in cases:
            with self.subTest(**parameters):
                example = make_random_qp(**parameters)
                validate_qp(example.qp_problem)
                self.assertEqual(example.n, parameters.get('n', 9))
                self.assertEqual(example.m, parameters.get('m', 13))
                for key in ('P', 'q', 'A', 'l', 'u'):
                    self.assertIs(example.qp_problem[key], getattr(example, key))
                self.assertTrue(np.isneginf(example.l).all())

                dense = example.P.toarray()
                np.testing.assert_allclose(dense, dense.T, atol=1e-12)
                eigenvalues, basis = np.linalg.eigh(dense)
                tolerance = 1e-10 * max(1.0, example.P_lambda_max_actual)
                self.assertGreaterEqual(eigenvalues[0], -tolerance)
                positive = eigenvalues[eigenvalues > tolerance]
                self.assertEqual(len(positive), example.P_rank)
                np.testing.assert_allclose(
                    eigenvalues, np.sort(example.P_eigenvalues), atol=tolerance)
                self.assertLessEqual(positive[-1], 5.0 + tolerance)
                self.assertLessEqual(positive[-1] / positive[0],
                                     parameters.get('P_cond_num', 20.0) + tolerance)
                self.assertAlmostEqual(example.P_lambda_max_actual, positive[-1])
                self.assertAlmostEqual(example.P_cond_num_actual,
                                       positive[-1] / positive[0])
                null_basis = basis[:, :example.n - example.P_rank]
                np.testing.assert_allclose(null_basis.T @ example.q, 0, atol=tolerance)

                offset = 0
                for b in example.P_block_sizes:
                    np.testing.assert_array_equal(dense[offset:offset + b, :offset], 0)
                    np.testing.assert_array_equal(dense[offset:offset + b, offset + b:], 0)
                    offset += b
                self.assertEqual(example.v.shape, (example.n,))
                slack = example.u - example.A @ example.v
                self.assertTrue(np.all(slack >= -tolerance))
                self.assertTrue(np.all(slack <= 1.0 + tolerance))
                for name in ('P', 'A'):
                    matrix = getattr(example, name)
                    self.assertTrue(sp.isspmatrix_csc(matrix))
                    self.assertTrue(np.all(matrix.data != 0))
                    density = np.count_nonzero(matrix.toarray()) / np.prod(matrix.shape)
                    self.assertEqual(getattr(example, name + '_density_actual'), density)
                    self.assertEqual(getattr(example, name + '_sparsity_actual'), 1 - density)
                if example.P_rank == 1 and example.n == 9:
                    # Eight zero blocks must not count toward the actual density.
                    self.assertEqual(example.P.nnz, 1)
                    self.assertEqual(example.P_density_actual, 1 / 81)

    def test_sampled_spectrum(self):
        rng = np.random.default_rng(2)
        positive = rng.uniform(5.0 / 20.0, 5.0, size=4)
        expected = np.concatenate([positive, np.zeros(5)])
        rng.shuffle(expected)
        example = make_random_qp(P_rank=4)
        np.testing.assert_array_equal(example.P_eigenvalues, expected)
        self.assertLess(example.P_lambda_max_actual, 5.0)
        self.assertLess(example.P_cond_num_actual, 20.0)

    def test_A_density_endpoints_and_cross_block_constraints(self):
        for density in (0.0, np.float32(0.4), 1.0):
            with self.subTest(density=density):
                example = make_random_qp(A_density=density)
                self.assertEqual(example.A.nnz, round(example.m * example.n * density))
                self.assertEqual(example.A_density_actual,
                                 example.A.nnz / (example.m * example.n))
                if density == 1.0:
                    self.assertTrue(np.all(example.A.toarray() != 0))

    def test_seed_reproducibility_without_global_rng_changes(self):
        before = np.random.get_state()
        first = make_random_qp(P_rank=4, seed=7)
        same = make_random_qp(P_rank=4, seed=7)
        different = make_random_qp(P_rank=4, seed=8)
        after = np.random.get_state()
        for old, new in zip(before, after):
            np.testing.assert_array_equal(old, new)
        for name in ('P', 'A'):
            for field in ('data', 'indices', 'indptr'):
                np.testing.assert_array_equal(getattr(getattr(first, name), field),
                                              getattr(getattr(same, name), field))
        for name in ('q', 'l', 'u', 'v', 'P_eigenvalues'):
            np.testing.assert_array_equal(getattr(first, name), getattr(same, name))
        self.assertGreater((first.P - different.P).nnz, 0)
        self.assertGreater((first.A - different.A).nnz, 0)

    def test_q_and_slack_scales(self):
        first = make_random_qp(P_rank=4)
        scaled = make_random_qp(P_rank=4, q_scale=2.5, slack_scale=4.0)
        zero_q = make_random_qp(P_rank=4, q_scale=0.0)
        np.testing.assert_allclose(scaled.q, 2.5 * first.q, atol=1e-12)
        np.testing.assert_array_equal(zero_q.q, 0)
        self.assertEqual((first.P - scaled.P).nnz, 0)
        self.assertEqual((first.A - scaled.A).nnz, 0)
        np.testing.assert_array_equal(first.v, scaled.v)
        np.testing.assert_allclose(scaled.u - scaled.A @ scaled.v,
                                   4.0 * (first.u - first.A @ first.v), atol=1e-12)

    def test_osqp_and_cvxpy_solutions(self):
        for parameters in ({}, dict(P_rank=4), dict(P_rank=4, A_density=0.0)):
            with self.subTest(**parameters):
                solve_and_check(make_random_qp(**parameters))

    def test_invalid_parameters(self):
        cases = (
            dict(n=0), dict(n=2.5), dict(m=0), dict(m=2.5),
            dict(P_block_sizes=[]), dict(P_block_sizes=[2, 4]),
            dict(P_block_sizes=[2, 7, 0]), dict(P_block_sizes=[2, -1, 8]),
            dict(P_block_sizes=[2, 3.5, 3.5]),
            dict(P_rank=0), dict(P_rank=10), dict(P_rank=2.5),
            dict(P_lambda_max=0), dict(P_lambda_max=-1), dict(P_lambda_max=np.inf),
            dict(P_cond_num=0.5), dict(P_cond_num=np.inf),
            dict(A_density=-0.1), dict(A_density=1.1), dict(A_density=np.nan),
            dict(q_scale=-1), dict(q_scale=np.inf),
            dict(slack_scale=0), dict(slack_scale=-1), dict(slack_scale=np.inf),
        )
        for parameters in cases:
            with self.subTest(**parameters), self.assertRaises(ValueError):
                make_random_qp(**parameters)


if __name__ == '__main__':
    unittest.main()
