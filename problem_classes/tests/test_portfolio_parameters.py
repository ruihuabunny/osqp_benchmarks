"""Checks for controlled sparse portfolio generation and its QP formulation."""

import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from test_generators import SETTINGS, solve_and_check
from problem_classes.portfolio import PortfolioExample


def make_portfolio(**overrides):
    parameters = dict(
        k=5, n=11, F_density=[0.5, 0.3], F_scale=1.5,
        F_block_sizes=[(7, 2), (4, 3)],
        D_spectrum=[4, 0, 0.25, 1, 2, 0, 8, 0.5, 0, 3, 1.5],
        mu_scale=1.2, gamma=2.5, seed=7)
    parameters.update(overrides)
    return PortfolioExample(**parameters)


class PortfolioParameterTests(unittest.TestCase):
    def test_rectangular_blocks_and_actual_statistics(self):
        example = make_portfolio()
        self.assertEqual(example.F[:7, 2:].nnz, 0)
        self.assertEqual(example.F[7:, :2].nnz, 0)
        self.assertEqual(example.F[:7, :2].nnz, 7)
        self.assertEqual(example.F[7:, 2:].nnz, 4)
        stats = example.generation_stats
        self.assertEqual(stats['density_F_blocks'], [0.5, 4 / 12])
        self.assertEqual(stats['density_F'], 11 / 55)
        self.assertNotEqual(stats['density_F'], np.mean(stats['density_F_blocks']))
        for name, matrix in (('F', example.F), ('D', example.D),
                             ('P', example.qp_problem['P']),
                             ('A', example.qp_problem['A'])):
            with self.subTest(matrix=name):
                self.assertTrue(sp.isspmatrix_csc(matrix))
                self.assertEqual(matrix.nnz, np.count_nonzero(matrix.data))
                density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
                self.assertEqual(stats[f'shape_{name}'], matrix.shape)
                self.assertEqual(stats[f'nnz_{name}'], matrix.nnz)
                self.assertEqual(stats[f'density_{name}'], density)
                self.assertEqual(stats[f'sparsity_{name}'], 1 - density)

    def test_spectrum_rank_and_condition_numbers(self):
        cases = (
            ([4, 0.25, 1, 2], 4, 16, 16),
            ([4, 0, 0.25, 0], 2, np.inf, 16),
            ([0, 0, 0, 0], 0, np.inf, 1),
            ([1e-20, 0, 2, 0], 2, np.inf, 2e20),
            ([4, 8, 2, 3], 4, 8, 8),
            ([0.5, 0.25, 0.125, 0.375], 4, 8, 8),
        )
        for values, rank, condition, positive_condition in cases:
            with self.subTest(spectrum=values):
                spectrum = np.array(values)
                example = make_portfolio(
                    k=2, n=4, F_block_sizes=[(4, 2)], F_density=[0.5],
                    D_spectrum=spectrum)
                np.testing.assert_array_equal(spectrum, values)
                np.testing.assert_array_equal(example.D.diagonal(), values)
                np.testing.assert_array_equal(
                    example.qp_problem['P'].diagonal(),
                    np.r_[2 * spectrum, [2, 2]])
                stats = example.generation_stats
                self.assertEqual(stats['D_rank'], rank)
                self.assertEqual(stats['P_rank'], rank + 2)
                self.assertEqual(stats['nnz_D'], rank)
                self.assertEqual(stats['nnz_P'], rank + 2)
                self.assertEqual(stats['P_condition_number'], condition)
                self.assertEqual(stats['P_positive_condition_number'], positive_condition)

    def test_reproducibility_and_scales(self):
        base = make_portfolio()
        same = make_portfolio()
        different = make_portfolio(seed=8)
        for key, value in base.qp_problem.items():
            if sp.issparse(value):
                for field in ('indptr', 'indices', 'data'):
                    np.testing.assert_array_equal(
                        getattr(value, field), getattr(same.qp_problem[key], field))
            else:
                np.testing.assert_array_equal(value, same.qp_problem[key])
        self.assertEqual(base.generation_stats, same.generation_stats)
        self.assertGreater((base.F - different.F).nnz, 0)
        self.assertFalse(np.array_equal(base.mu, different.mu))
        np.testing.assert_array_equal(base.D.diagonal(), different.D.diagonal())

        scaled = make_portfolio(F_scale=3.0, mu_scale=3.6, gamma=5.0)
        np.testing.assert_array_equal(scaled.F.indptr, base.F.indptr)
        np.testing.assert_array_equal(scaled.F.indices, base.F.indices)
        np.testing.assert_allclose(scaled.F.data, 2 * base.F.data)
        np.testing.assert_allclose(scaled.mu, 3 * base.mu)
        np.testing.assert_array_equal(scaled.D.diagonal(), base.D.diagonal())
        self.assertEqual(scaled.gamma, 5.0)
        np.testing.assert_allclose(scaled.qp_problem['q'], 1.5 * base.qp_problem['q'])
        np.testing.assert_array_equal(make_portfolio(mu_scale=0).mu, np.zeros(11))

    def test_blocks_share_a_random_stream(self):
        example = make_portfolio(
            k=6, n=12, F_block_sizes=[(6, 3), (6, 3)],
            F_density=[0.5, 0.5], D_spectrum=np.ones(12))
        self.assertGreater((example.F[:6, :3] - example.F[6:, 3:]).nnz, 0)

    def test_zero_and_full_density_and_zero_scale(self):
        for scale in (0.0, 2.0):
            with self.subTest(F_scale=scale):
                example = make_portfolio(
                    F_density=[0, 1], F_scale=scale, D_spectrum=np.zeros(11))
                count = 0 if scale == 0 else 12
                self.assertEqual(example.F.nnz, count)
                self.assertEqual(np.count_nonzero(example.F.data), count)
                self.assertEqual(example.generation_stats['density_F_blocks'],
                                 [0, 0 if scale == 0 else 1])
                self.assertEqual(example.generation_stats['density_F'], count / 55)
                self.assertEqual(example.D.nnz, 0)
                self.assertEqual(example.qp_problem['P'].nnz, 5)
                self.assertEqual(example.qp_problem['A'].nnz, 27 + count)

    def test_qp_constraints_bounds_and_original_objective(self):
        example = make_portfolio()
        n, k = example.n, example.k
        qp = example.qp_problem
        self.assertEqual((qp['n'], qp['m']), (n + k, n + k + 1))
        np.testing.assert_array_equal(qp['q'], np.r_[-example.mu / example.gamma, np.zeros(k)])
        np.testing.assert_array_equal(qp['l'], np.r_[1, np.zeros(k + n)])
        np.testing.assert_array_equal(qp['u'], np.r_[1, np.zeros(k), np.ones(n)])
        np.testing.assert_array_equal(qp['bounds_idx'], np.arange(n))
        np.testing.assert_array_equal(qp['lx'], np.r_[np.zeros(n), np.full(k, -np.inf)])
        np.testing.assert_array_equal(qp['ux'], np.r_[np.ones(n), np.full(k, np.inf)])
        x = np.arange(1, n + 1, dtype=float)
        x /= x.sum()
        y = example.F.T @ x
        z = np.r_[x, y]
        np.testing.assert_allclose(qp['A'] @ z, np.r_[1, np.zeros(k), x], atol=1e-14)
        self.assertAlmostEqual(
            0.5 * z @ (qp['P'] @ z) + qp['q'] @ z,
            x @ (example.D @ x) + y @ y - example.mu @ x / example.gamma)

        result = solve_and_check(example)
        assets = cp.Variable(n)
        original = cp.Problem(cp.Minimize(
            cp.quad_form(assets, example.D) + cp.sum_squares(example.F.T @ assets)
            - example.mu @ assets / example.gamma),
            [cp.sum(assets) == 1, assets >= 0, assets <= 1])
        original.solve(solver=cp.OSQP, **SETTINGS)
        self.assertEqual(original.status, cp.OPTIMAL)
        self.assertAlmostEqual(result['osqp_objective'], original.value,
                               delta=5e-6 * (1 + abs(original.value)))

    def test_default_asset_count_and_large_sparse_generation(self):
        default = make_portfolio(k=2, n=None, F_block_sizes=[(200, 2)],
                                 F_density=[0.2], D_spectrum=np.ones(200))
        self.assertEqual(default.n, 200)
        self.assertEqual(default.F.shape, (200, 2))
        self.assertEqual(default.F.nnz, 80)
        large = make_portfolio(
            k=100, n=10000, F_block_sizes=[(6000, 60), (4000, 40)],
            F_density=[0.005, 0.01], D_spectrum=np.tile([0, 0.25, 1, 4], 2500))
        qp = large.qp_problem
        self.assertEqual(large.F.nnz, 3400)
        self.assertEqual((qp['n'], qp['m']), (10100, 10101))
        self.assertEqual(qp['P'].nnz, 7600)
        self.assertEqual(qp['A'].nnz, 23500)
        for matrix in (large.F, large.D, qp['P'], qp['A']):
            self.assertTrue(sp.isspmatrix_csc(matrix))

    def test_invalid_parameters(self):
        cases = (
            dict(F_density=[0.5]), dict(F_density=[-0.1, 0.5]),
            dict(F_density=[0.5, 1.1]), dict(F_density=[np.nan, 0.5]),
            dict(F_block_sizes=[(6, 2), (4, 3)]),
            dict(F_block_sizes=[(7, 2), (4, 2)]),
            dict(D_spectrum=np.ones(10)), dict(D_spectrum=np.ones((11, 1))),
            dict(D_spectrum=[-1] + [1] * 10),
            dict(D_spectrum=[np.nan] + [1] * 10),
            dict(gamma=0), dict(gamma=-1), dict(gamma=np.inf),
        )
        for overrides in cases:
            with self.subTest(**overrides):
                with self.assertRaises(ValueError):
                    make_portfolio(**overrides)


if __name__ == '__main__':
    unittest.main()
