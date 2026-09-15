import numpy as np
import scipy.sparse as spa
import cvxpy


class HuberExample(object):
    '''
    Huber regression with a block-diagonal feature matrix and controlled spectrum.

    Each block is U_j @ diag(s_j) @ V_j.T, with orthonormal columns in
    U_j and V_j. Only individual blocks are dense; the assembled Ad is CSC.
    The regression problem can be decomposed into independent block problems.
    Dense work is limited to one block at a time, so block sizes also control
    generation memory and cost. No full SVD is used during generation.
    '''
    def __init__(self, n, m, num_blocks, block_sizes, r,
                 max_singular_val, max_cond_num, delta,
                 sigma, outlier_fraction, outlier_scale, seed=1):
        '''
        Generate problem in QP format and CVXPY format.

        n, m: positive integer feature and sample counts
        num_blocks: positive integer number of blocks
        block_sizes: list of (m_j, n_j, r_j), one triple per block, summing
            to (m, n, r); 1 <= r_j <= min(m_j, n_j). U_j has shape
            (m_j, r_j), while V_j has shape (n_j, r_j).
        r: positive target rank of Ad, distinct from the QP auxiliary vector r
        max_singular_val: finite positive upper bound S on singular values
        max_cond_num: finite upper bound C >= 1 on the ratio of the largest
            to smallest POSITIVE singular value. For rank-deficient Ad this
            is not the usual condition number including zero singular values.
        delta: finite positive Huber threshold; loss is 0.5*t**2 for
            abs(t) <= delta, otherwise delta*abs(t) - 0.5*delta**2
        sigma: finite nonnegative standard deviation of ordinary Gaussian noise
        outlier_fraction: probability in [0, 1] of outlier noise for each sample
        outlier_scale: finite nonnegative scale of Uniform(0, 1) outlier noise
        seed: local RNG seed controlling all random steps

        The global positive spectrum is S * C**(-i/(r-1)), i=0,...,r-1,
        allocated to blocks in order. For r=1 the only singular value is S
        and the positive-singular-value ratio is 1. These are exact-arithmetic
        targets; very small singular values may be lost to floating-point error.
        x_true is N(0, I)/sqrt(n), a data-generation parameter, not a known
        exact minimizer of the noisy Huber problem.

        generation_stats records nnz_U, nnz_V, nnz_Ad and density_/sparsity_
        for U, V and Ad. Counts are actual nonzeros, without thresholding.
        The implicit global U and V have shapes (m, r) and (n, r):
            density_U = sum(count_nonzero(U_j)) / (m*r)
            density_V = sum(count_nonzero(V_j)) / (n*r)
            density_Ad = count_nonzero(Ad) / (m*n)
        Each sparsity is 1 - density. For dense blocks the numerators are
        sum(m_j*r_j), sum(n_j*r_j), and sum(m_j*n_j), respectively.
        The QP has n + 3*m variables and 3*m constraint rows.

        Example: HuberExample(
            n=20, m=100, num_blocks=2, block_sizes=[(60, 12, 8), (40, 8, 4)],
            r=12, max_singular_val=10.0, max_cond_num=100.0, delta=1.5,
            sigma=0.5, outlier_fraction=0.05, outlier_scale=10.0, seed=1)
        '''
        for name, value in (('n', n), ('m', m), ('num_blocks', num_blocks),
                            ('r', r)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if len(block_sizes) != num_blocks:
            raise ValueError('block_sizes must contain num_blocks triples')
        self.block_sizes = []
        for block in block_sizes:
            if len(block) != 3 or any(
                    not isinstance(value, (int, np.integer)) or value < 1
                    for value in block):
                raise ValueError('each block must contain positive integers (m_j, n_j, r_j)')
            m_j, n_j, r_j = map(int, block)
            if r_j > min(m_j, n_j):
                raise ValueError('each block must satisfy r_j <= min(m_j, n_j)')
            self.block_sizes.append((m_j, n_j, r_j))
        totals = tuple(sum(block[i] for block in self.block_sizes) for i in range(3))
        if totals != (m, n, r):
            raise ValueError('block dimensions and ranks must sum to (m, n, r)')
        for name, value in (('max_singular_val', max_singular_val), ('delta', delta)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(max_cond_num) or max_cond_num < 1:
            raise ValueError('max_cond_num must be finite and at least 1')
        for name, value in (('sigma', sigma), ('outlier_scale', outlier_scale)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= outlier_fraction <= 1:
            raise ValueError('outlier_fraction must be in [0, 1]')

        self.n = int(n)               # Number of features
        self.m = int(m)               # Number of data-points
        self.num_blocks = int(num_blocks)
        self.r = int(r)
        self.max_singular_val = float(max_singular_val)
        self.max_cond_num = float(max_cond_num)
        self.delta = float(delta)
        self.sigma = float(sigma)
        self.outlier_fraction = float(outlier_fraction)
        self.outlier_scale = float(outlier_scale)
        rng = np.random.default_rng(seed)

        # One global spectrum: max_cond_num bounds max(s_positive)/min(s_positive).
        if self.r == 1:
            singular_values = np.array([self.max_singular_val])
        else:
            singular_values = self.max_singular_val * self.max_cond_num ** (
                -np.arange(self.r) / (self.r - 1))
        blocks = []
        offset = 0
        nnz_u = nnz_v = 0
        for m_j, n_j, r_j in self.block_sizes:
            U_j = np.linalg.qr(rng.standard_normal((m_j, r_j)), mode='reduced')[0]
            V_j = np.linalg.qr(rng.standard_normal((n_j, r_j)), mode='reduced')[0]
            nnz_u += np.count_nonzero(U_j)
            nnz_v += np.count_nonzero(V_j)
            s_j = singular_values[offset:offset + r_j]
            blocks.append(spa.csc_matrix((U_j * s_j) @ V_j.T))
            offset += r_j
            del U_j, V_j
        self.Ad = spa.block_diag(blocks, format='csc')
        del blocks

        # block_diag has no duplicate entries. Count nonzero values, not stored
        # entries (nnz), without modifying Ad or discarding small coefficients.
        nnz_ad = np.count_nonzero(self.Ad.data)
        self.generation_stats = {}
        for name, count, size in (('U', nnz_u, self.m * self.r),
                                  ('V', nnz_v, self.n * self.r),
                                  ('Ad', nnz_ad, self.m * self.n)):
            density = count / size
            self.generation_stats[f'nnz_{name}'] = int(count)
            self.generation_stats[f'density_{name}'] = density
            self.generation_stats[f'sparsity_{name}'] = 1.0 - density

        self.x_true = rng.standard_normal(self.n) / np.sqrt(self.n)
        is_outlier = rng.random(self.m) < self.outlier_fraction
        ordinary_noise = self.sigma * rng.standard_normal(self.m)
        outlier_noise = self.outlier_scale * rng.random(self.m)
        noise = np.where(is_outlier, outlier_noise, ordinary_noise)
        self.bd = self.Ad @ self.x_true + noise

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem, self.cvxpy_variables = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Huber'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        # Construct the problem
        #       minimize    1/2 z.T * z + delta * np.ones(m).T * (r + s)
        #       subject to  Ax - b - z = r - s
        #                   r >= 0
        #                   s >= 0
        # The problem reformulation follows from Eq. (24) of the following paper:
        # https://doi.org/10.1109/34.877518
        # x_solver = (x, z, r, s)
        Im = spa.eye(self.m)
        P = spa.block_diag((spa.csc_matrix((self.n, self.n)), Im,
                            spa.csc_matrix((2*self.m, 2*self.m))), format='csc')
        q = np.hstack([np.zeros(self.n + self.m),
                       np.full(2*self.m, self.delta)])
        A = spa.bmat([[self.Ad, -Im,   -Im,   Im],
                      [None,     None,  Im,   None],
                      [None,     None,  None, Im]], format='csc')
        l = np.hstack([self.bd, np.zeros(2*self.m)])
        u = np.hstack([self.bd, np.inf*np.ones(2*self.m)])

        # Constraints without bounds
        A_nobounds = spa.hstack([self.Ad, -Im, -Im, Im], format='csc')
        l_nobounds = self.bd
        u_nobounds = self.bd

        # Bounds
        lx = np.hstack([-np.inf * np.ones(self.n + self.m),
                        np.zeros(2*self.m)])
        ux = np.inf*np.ones(self.n + 3*self.m)
        bounds_idx = np.arange(self.n + self.m, self.n + 3*self.m)

        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]
        problem['A_nobounds'] = A_nobounds
        problem['l_nobounds'] = l_nobounds
        problem['u_nobounds'] = u_nobounds
        problem['bounds_idx'] = bounds_idx
        problem['lx'] = lx
        problem['ux'] = ux

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        # Construct the problem
        #       minimize    1/2 z.T * z + delta * np.ones(m).T * (r + s)
        #       subject to  Ax - b - z = r - s
        #                   r >= 0
        #                   s >= 0
        # The problem reformulation follows from Eq. (24) of the following paper:
        # https://doi.org/10.1109/34.877518
        x = cvxpy.Variable(self.n)
        z = cvxpy.Variable(self.m)
        r = cvxpy.Variable(self.m)
        s = cvxpy.Variable(self.m)

        objective = cvxpy.Minimize(.5 * cvxpy.sum_squares(z)
                                   + self.delta * cvxpy.sum(r + s))
        constraints = [self.Ad@x - self.bd - z == r - s,
                       r >= 0, s >= 0]
        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, z, r, s)

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        (x_cvx, z_cvx, r_cvx, s_cvx) = self.cvxpy_variables
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((x_cvx.value,
                            z_cvx.value,
                            r_cvx.value,
                            s_cvx.value))

        # dual solution
        y = np.concatenate((constraints[0].dual_value,
                            -constraints[1].dual_value,
                            -constraints[2].dual_value))

        return x, y
