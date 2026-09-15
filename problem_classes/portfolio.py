import numpy as np
import scipy.sparse as spa
import cvxpy


class PortfolioExample(object):
    '''
    Portfolio QP example
    '''
    def __init__(self, k, F_density, F_scale,
                 F_block_sizes, D_spectrum,
                 mu_scale, gamma, seed=1, n=None):
        '''
        Generate a factor-model portfolio problem in QP and CVXPY format.

        k: number of risk factors
        F_density: one target nonzero density in [0, 1] per F block;
            entries outside the diagonal blocks are zero
        F_scale: multiplier of standard-normal values at sampled F positions
        F_block_sizes: list of rectangular (rows, columns) block sizes;
            row counts must sum to n and column counts to k
        D_spectrum: length-n array of finite nonnegative diagonal entries of D,
            used in the supplied order without scaling; zeros control its rank
        mu_scale: multiplier of the standard-normal expected-return vector mu
        gamma: finite positive risk-aversion coefficient
        seed: NumPy random seed shared by all blocks and mu
        n: number of assets; defaults to 100*k

        The auxiliary variable y = F.T @ x gives n+k QP variables and n+k+1
        constraint rows. F is assembled directly as a sparse CSC matrix.

        generation_stats records the initial matrices: density_F_blocks holds
        actual block densities; shape_*, nnz_*, density_* and sparsity_* describe
        F, D, P and A after removing explicit zeros. Density is nnz/(rows*columns)
        and sparsity is 1-density. D_rank counts strictly positive entries of
        D_spectrum and P_rank is D_rank+k. P_condition_number is infinite for
        singular P; P_positive_condition_number is the largest/smallest strictly
        positive diagonal entry of P. Both are computed directly from P.
        '''
        # Set random seed
        np.random.seed(seed)

        self.k = int(k)               # Number of factors
        if n is None:                 # Number of assets
            self.n = int(k * 100)
        else:
            self.n = int(n)

        F_density = np.asarray(F_density)
        D_spectrum = np.asarray(D_spectrum)
        if F_density.shape != (len(F_block_sizes),):
            raise ValueError('F_density must contain one value per F block')
        if not np.all((0 <= F_density) & (F_density <= 1)):
            raise ValueError('F_density values must be in [0, 1]')
        if (sum(rows for rows, _ in F_block_sizes) != self.n or
                sum(columns for _, columns in F_block_sizes) != self.k):
            raise ValueError('F block row and column counts must sum to n and k')
        if D_spectrum.shape != (self.n,):
            raise ValueError('D_spectrum must be a length-n array')
        if not np.all(np.isfinite(D_spectrum) & (D_spectrum >= 0)):
            raise ValueError('D_spectrum entries must be finite and nonnegative')
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('gamma must be finite and positive')

        # Generate sparse blocks using the same random stream throughout.
        blocks = []
        block_densities = []
        for (rows, columns), density in zip(F_block_sizes, F_density):
            block = spa.random(
                rows, columns, density=density, format='csc',
                data_rvs=lambda size: F_scale * np.random.standard_normal(size))
            block.eliminate_zeros()
            blocks.append(block)
            block_densities.append(block.nnz / (rows * columns))
        self.F = spa.block_diag(blocks, format='csc')
        del blocks
        self.D = spa.diags(D_spectrum, format='csc', dtype=float)
        self.mu = mu_scale * np.random.standard_normal(self.n)
        self.gamma = gamma

        self.qp_problem = self._generate_qp_problem()

        self.generation_stats = {'density_F_blocks': block_densities}
        for name, matrix in (('F', self.F), ('D', self.D),
                             ('P', self.qp_problem['P']),
                             ('A', self.qp_problem['A'])):
            matrix.eliminate_zeros()
            density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
            self.generation_stats[f'shape_{name}'] = matrix.shape
            self.generation_stats[f'nnz_{name}'] = int(matrix.nnz)
            self.generation_stats[f'density_{name}'] = density
            self.generation_stats[f'sparsity_{name}'] = 1.0 - density

        D_rank = int(np.count_nonzero(D_spectrum > 0))
        diagonal = self.qp_problem['P'].diagonal()
        positive = diagonal[diagonal > 0]
        self.generation_stats['D_rank'] = D_rank
        self.generation_stats['P_rank'] = D_rank + self.k
        self.generation_stats['P_condition_number'] = (
            float(diagonal.max() / diagonal.min())
            if np.all(diagonal > 0) else np.inf)
        self.generation_stats['P_positive_condition_number'] = float(
            positive.max() / positive.min())

        self.cvxpy_problem, self.cvxpy_param = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Portfolio'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	x' D x + y' I y - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   F' x = y
        #                   0 <= x <= 1
        P = spa.block_diag((2 * self.D, 2 * spa.eye(self.k)), format='csc')
        q = np.append(- self.mu / self.gamma, np.zeros(self.k))
        A = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                           spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                spa.hstack((spa.eye(self.n), spa.csc_matrix((self.n, self.k))))
            ]).tocsc()
        l = np.hstack([1., np.zeros(self.k), np.zeros(self.n)])
        u = np.hstack([1., np.zeros(self.k), np.ones(self.n)])

        # Constraints without bounds
        A_nobounds = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                            spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                ]).tocsc()
        l_nobounds = np.hstack([1., np.zeros(self.k)])
        u_nobounds = np.hstack([1., np.zeros(self.k)])
        bounds_idx = np.arange(self.n)

        # Separate bounds
        lx = np.hstack([np.zeros(self.n), -np.inf * np.ones(self.k)])
        ux = np.hstack([np.ones(self.n), np.inf * np.ones(self.k)])

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

        x = cvxpy.Variable(self.n)
        y = cvxpy.Variable(self.k)

        # Create parameters m
        mu = cvxpy.Parameter(self.n)
        mu.value = self.mu

        objective = cvxpy.Minimize(cvxpy.quad_form(x, self.D) +
                                   cvxpy.quad_form(y, spa.eye(self.k)) +
                                   - 1 / self.gamma * (mu.T @ x))
        constraints = [cvxpy.sum(x) == 1,
                       self.F.T @ x == y,
                       0 <= x, x <= 1]
        problem = cvxpy.Problem(objective, constraints)

        return problem, mu

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((variables[0].value,
                            variables[1].value))

        # dual solution
        y = np.concatenate(([constraints[0].dual_value],
                            constraints[1].dual_value,
                            constraints[3].dual_value -
                            constraints[2].dual_value))

        return x, y

    def update_parameters(self, mu, F=None, D=None):
        """
        Update problem parameters with new mu, F, D
        """

        # Update internal parameters
        self.mu = mu
        if F is not None:
            if F.shape == self.F.shape and \
                    all(F.indptr == self.F.indptr) and \
                    all(F.indices == self.F.indices):
                # Check if F has same sparsity pattern as self.D
                self.F = F
            else:
                raise ValueError("F sparsity pattern changed")
        if D is not None:
            if D.shape == self.D.shape and \
                    all(D.indptr == self.D.indptr) and \
                    all(D.indices == self.D.indices):
                # Check if D has same sparsity pattern as self.D
                self.D = D
            else:
                raise ValueError("D sparsity pattern changed")

        # Update parameters in QP problem
        if F is None and D is None:
            # Update only q
            self.qp_problem['q'] = np.append(- self.mu / self.gamma,
                                             np.zeros(self.k))
            # Update parameter in CVXPY problem
            self.cvxpy_param.value = self.mu
        else:
            # Generate problem from scratch
            self.qp_problem = self._generate_qp_problem()
            self.cvxpy_problem, self.cvxpy_param = \
                self._generate_cvxpy_problem()
