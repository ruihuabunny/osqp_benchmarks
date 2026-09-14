import numpy as np
import scipy.sparse as spa
import cvxpy


class EqQPExample(object):
    '''
    Equality constrained QP example
    '''
    def __init__(self, n, m=None, lower_density=0.15, seed=1,
                 max_spectrum=None, max_cond_num=None, r=None):
        '''
        Generate problem in QP format and CVXPY format
        n: dimension of the matrix self.P
        m: number of constraints, from 1 to n; defaults to n // 2
        lower_density: nonzero density of the lower matrix and constraint
            matrix, in (0, 1]; this is not the density of the assembled P
        seed: random seed
        r: rank of self.P, from 0 to n; defaults to n
        max_spectrum: upper bound on the largest eigenvalue; None leaves it uncapped
        max_cond_num: upper bound on the ratio of largest to smallest positive
            eigenvalue; None leaves it uncapped, and r=0 has no such ratio

        Bounds are conservative and need not be attained. A spectrum bound must
        be finite and positive for r>0 (zero is allowed for r=0). A condition
        bound must be finite and at least 1, and requires r>0.
        '''
        self.n = int(n)
        m = self.n // 2 if m is None else m
        if not isinstance(m, (int, np.integer)) or not 1 <= m <= self.n:
            raise ValueError('m must be a positive integer smaller than or equal to n')
        self.m = int(m)
        self.r = self.n if r is None else r
        self.lower_density = lower_density
        if not 0 < lower_density <= 1:
            raise ValueError('lower_density must be in (0, 1]')
        if not isinstance(self.r, (int, np.integer)) or not 0 <= self.r <= self.n:
            raise ValueError('r must be an integer between 0 and n')
        if max_spectrum is not None:
            if not np.isfinite(max_spectrum) or max_spectrum < 0:
                raise ValueError('max_spectrum must be finite and nonnegative')
            if self.r > 0 and max_spectrum == 0:
                raise ValueError('max_spectrum must be positive when r > 0')
        if max_cond_num is not None:
            if not np.isfinite(max_cond_num) or max_cond_num < 1:
                raise ValueError('max_cond_num must be finite and at least 1')
            if self.r == 0:
                raise ValueError('max_cond_num is undefined when r is 0')
        rng = np.random.default_rng(seed)

        # The identity block makes G full column rank, including sparse cases.
        lower = spa.random(self.n - self.r, self.r, density=self.lower_density,
                           data_rvs=rng.standard_normal, random_state=rng,
                           format='csc')
        lower_norm_sq = lower.data @ lower.data
        weight_min = 0.1
        if max_cond_num is not None:
            # G.T @ G = I + lower.T @ lower. Split the condition bound
            # between this Gram matrix and the positive diagonal weights.
            condition_scale = np.sqrt(max_cond_num)
            weight_min = 1.0 / condition_scale
            if lower_norm_sq > condition_scale - 1.0:
                lower *= np.sqrt((condition_scale - 1.0) / lower_norm_sq)
                lower.eliminate_zeros()
        G = spa.vstack([spa.eye(self.r, format='csc'), lower], format='csc')
        G = G[rng.permutation(self.n), :]
        weights = rng.uniform(weight_min, 1.0, self.r)
        Lambda = spa.diags(weights,
                           shape=(self.r, self.r), format='csc')
        self.P = (G @ Lambda @ G.T).tocsc()
        if max_spectrum is not None and self.r > 0:
            if max_cond_num is None:
                spectrum_bound = weights.max() * (1.0 + lower_norm_sq)
                self.P *= min(1.0, max_spectrum / spectrum_bound)
            else:
                # Positive eigenvalues then lie in [max_spectrum/K, max_spectrum].
                self.P *= max_spectrum / condition_scale
        self.A = spa.random(self.m, self.n, density=self.lower_density,
                            data_rvs=rng.standard_normal, random_state=rng,
                            format='csc')

        # Plant an optimal primal/dual pair to keep rank-deficient QPs bounded.
        x_sol = rng.standard_normal(self.n)
        y_sol = rng.standard_normal(self.m)
        self.l = self.A@x_sol
        self.u = np.copy(self.l)
        self.q = -self.P @ x_sol - self.A.T @ y_sol

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Eq QP'

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
        # P is PSD by construction; numerical checks can fail at zero eigenvalues.
        objective = (.5 * cvxpy.quad_form(x_var, cvxpy.psd_wrap(self.P))
                     + self.q @ x_var)
        constraints = [self.A @ x_var == self.u]
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
        y = constraints[0].dual_value

        return x, y
