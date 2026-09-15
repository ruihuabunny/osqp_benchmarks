import numpy as np
import scipy.sparse as spa
import cvxpy


class RandomQPExample(object):
    '''
    Random QP example
    '''
    def __init__(
        self,
        n,
        m,
        P_block_sizes,
        P_rank,
        P_lambda_max,
        P_cond_num,
        A_density,
        q_scale,
        slack_scale,
        seed=1,
    ):
        '''
        Generate min 0.5*x.T@P@x + q.T@x subject to A@x <= u.

        n, m: independent positive integer variable and constraint counts
        P_block_sizes: positive integer block sizes summing to n; only these
            diagonal blocks can be nonzero, and dense work stays within a block
        P_rank: global rank of P, an integer from 1 to n
        P_lambda_max: finite positive upper bound on the largest eigenvalue
        P_cond_num: finite upper bound >= 1 on the nonzero spectral condition
            number, the largest/smallest positive eigenvalue ratio; the ordinary
            condition number is infinite when P is rank deficient
        A_density: target nonzero density in [0, 1] of the global m-by-n A;
            standard-normal entries can couple different P blocks
        q_scale: finite nonnegative Gaussian scale; q lies in range(P), making
            the objective bounded below even when P is rank deficient;
            for full rank, q has distribution N(0, q_scale**2 * I)
        slack_scale: finite positive scale of the algebraic slack u - A@v at
            the generated feasible point v; it is neither a Euclidean distance
            nor a control on the number of active constraints at an optimum
        seed: local RNG seed shared by all random steps

        Positive eigenvalues are sampled uniformly in
        [P_lambda_max/P_cond_num, P_lambda_max], padded with zeros and shuffled
        across blocks. Both spectral bounds need not be attained. P_eigenvalues
        stores this generated spectrum; P_lambda_max_actual and P_cond_num_actual
        are its maximum and nonzero spectral condition number, computed without
        a matrix eigendecomposition. Rank and spectrum are exact-arithmetic
        targets, subject to floating-point roundoff in block construction.

        P_density_actual and A_density_actual use the final CSC matrices after
        removing explicit zeros; P_sparsity_actual and A_sparsity_actual are
        their complements. self.v is a construction point, not a known optimum.
        The OSQP lower bound l is -inf for every constraint.
        '''
        for name, value in (('n', n), ('m', m)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        P_block_sizes = list(P_block_sizes)
        if any(not isinstance(b, (int, np.integer)) or b < 1
               for b in P_block_sizes):
            raise ValueError('P_block_sizes must contain positive integers')
        if sum(P_block_sizes) != n:
            raise ValueError('P_block_sizes must sum to n')
        if not isinstance(P_rank, (int, np.integer)) or not 1 <= P_rank <= n:
            raise ValueError('P_rank must be an integer between 1 and n')
        for name, value in (('P_lambda_max', P_lambda_max),
                            ('slack_scale', slack_scale)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(P_cond_num) or P_cond_num < 1:
            raise ValueError('P_cond_num must be finite and at least 1')
        if not 0 <= A_density <= 1:
            raise ValueError('A_density must be in [0, 1]')
        if not np.isfinite(q_scale) or q_scale < 0:
            raise ValueError('q_scale must be finite and nonnegative')

        self.n = int(n)
        self.m = int(m)
        self.P_block_sizes = [int(b) for b in P_block_sizes]
        self.P_rank = int(P_rank)
        rng = np.random.default_rng(seed)

        positive_eigenvalues = rng.uniform(
            P_lambda_max / P_cond_num, P_lambda_max, size=self.P_rank)
        self.P_eigenvalues = np.concatenate([
            positive_eigenvalues, np.zeros(self.n - self.P_rank)])
        rng.shuffle(self.P_eigenvalues)
        self.P_lambda_max_actual = float(positive_eigenvalues.max())
        self.P_cond_num_actual = float(
            positive_eigenvalues.max() / positive_eigenvalues.min())

        blocks, q_blocks = [], []
        offset = 0
        for b in self.P_block_sizes:
            block_eigenvalues = self.P_eigenvalues[offset:offset + b]
            Q_block = np.linalg.qr(rng.standard_normal((b, b)))[0]
            blocks.append(spa.csc_matrix(
                (Q_block * block_eigenvalues) @ Q_block.T))
            positive = block_eigenvalues > 0
            q_blocks.append(q_scale * (
                Q_block[:, positive]
                @ rng.standard_normal(np.count_nonzero(positive))))
            offset += b
            del Q_block
        self.P = spa.block_diag(blocks, format='csc')
        self.q = np.concatenate(q_blocks)
        del blocks, q_blocks

        self.A = spa.random(self.m, self.n, density=A_density,
                            data_rvs=rng.standard_normal, random_state=rng,
                            format='csc')
        self.v = rng.standard_normal(self.n)
        delta = rng.random(self.m)
        self.u = self.A @ self.v + slack_scale * delta
        self.l = np.full(self.m, -np.inf)

        self.P.eliminate_zeros()
        self.A.eliminate_zeros()
        self.P_density_actual = self.P.nnz / (self.n * self.n)
        self.A_density_actual = self.A.nnz / (self.m * self.n)
        self.P_sparsity_actual = 1.0 - self.P_density_actual
        self.A_sparsity_actual = 1.0 - self.A_density_actual

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random QP'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        # P is PSD by construction, including when some eigenvalues are zero.
        objective = (.5 * cvxpy.quad_form(x_var, cvxpy.psd_wrap(self.P))
                     + self.q @ x_var)
        constraints = [self.A @ x_var <= self.u, self.A @ x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = constraints[0].dual_value - constraints[1].dual_value

        return x, y
