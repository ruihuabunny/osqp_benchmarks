import numpy as np
import scipy.sparse as spa
import cvxpy


class LassoExample(object):
    '''
    Lasso QP example
    '''
    def __init__(self, n, seed=1, m=None, density=0.15,
                 data_scale=1.0, lambda_ratio=0.1):
        '''
        Generate problem in QP format and CVXPY format
        n: number of features
        seed: random seed; retains the original second positional argument
        m: number of data points; defaults to 100 * n
        density: nonzero density of the data matrix Ad, in (0, 1]
        data_scale: positive multiplier applied to both Ad and bd, including
            observation noise; preserves the signal-to-noise ratio
        lambda_ratio: nonnegative lambda / lambda_max, where
            lambda_max = 2 * ||Ad.T @ bd||_inf is the zero-solution threshold
            for ||Ad @ x - bd||_2^2 + lambda * ||x||_1

        The default ratio 0.1 preserves the previous regularization strength
        relative to the zero-solution threshold. Ratios >= 1 admit x = 0.
        The final QP has m + 2*n variables and m + 2*n constraints.
        '''
        self.n = int(n)               # Number of features
        m = 100 * self.n if m is None else m
        if self.n < 1:
            raise ValueError('n must be positive')
        if not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError('m must be a positive integer')
        if not 0 < density <= 1:
            raise ValueError('density must be in (0, 1]')
        if not np.isfinite(data_scale) or data_scale <= 0:
            raise ValueError('data_scale must be finite and positive')
        if not np.isfinite(lambda_ratio) or lambda_ratio < 0:
            raise ValueError('lambda_ratio must be finite and nonnegative')
        self.m = int(m)
        self.density = density
        self.data_scale = data_scale
        rng = np.random.default_rng(seed)

        self.Ad = spa.random(self.m, self.n, density=self.density,
                             data_rvs=rng.standard_normal, random_state=rng,
                             format='csc')
        self.x_true = ((rng.random(self.n) > 0.5)
                       * rng.standard_normal(self.n)) / np.sqrt(self.n)
        self.bd = self.Ad @ self.x_true + rng.standard_normal(self.m)
        self.Ad *= self.data_scale
        self.bd *= self.data_scale
        self.lambda_max = 2 * np.linalg.norm(self.Ad.T @ self.bd, np.inf)
        self.lambda_param = lambda_ratio * self.lambda_max

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem, self.cvxpy_variables, self.cvxpy_param = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Lasso'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	y' * y + lambda * 1' * t
        #       subject to  y = Ax - b
        #                   -t <= x <= t
        P = spa.block_diag((spa.csc_matrix((self.n, self.n)),
                            2*spa.eye(self.m),
                            spa.csc_matrix((self.n, self.n))), format='csc')
        q = np.append(np.zeros(self.m + self.n),
                      self.lambda_param * np.ones(self.n))
        In = spa.eye(self.n)
        Onm = spa.csc_matrix((self.n, self.m))
        A = spa.vstack([spa.hstack([self.Ad, -spa.eye(self.m),
                                    spa.csc_matrix((self.m, self.n))]),
                        spa.hstack([In, Onm, -In]),
                        spa.hstack([In, Onm, In])]).tocsc()
        l = np.hstack([self.bd, -np.inf * np.ones(self.n), np.zeros(self.n)])
        u = np.hstack([self.bd, np.zeros(self.n), np.inf * np.ones(self.n)])

        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''

        x = cvxpy.Variable(self.n)
        y = cvxpy.Variable(self.m)
        t = cvxpy.Variable(self.n)

        # Use the parameter in the objective so update_lambda updates both forms.
        lambda_cvxpy = cvxpy.Parameter(nonneg=True)
        lambda_cvxpy.value = self.lambda_param

        objective = cvxpy.Minimize(cvxpy.quad_form(y, spa.eye(self.m))
                                   + lambda_cvxpy * cvxpy.sum(t))
        constraints = [y == self.Ad @ x - self.bd,
                       -t <= x, x <= t]
        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, y, t), lambda_cvxpy

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        (x_cvx, y_cvx, t_cvx) = self.cvxpy_variables
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((x_cvx.value,
                            y_cvx.value,
                            t_cvx.value))

        # dual solution
        y = np.concatenate((-constraints[0].dual_value,
                            constraints[2].dual_value,
                            -constraints[1].dual_value))

        return x, y

    def update_lambda(self, lambda_new):
        """
        Update lambda value in inner problems
        """
        # Validate the parameter before changing the QP data.
        self.cvxpy_param.value = lambda_new

        # Update internal lambda parameter
        self.lambda_param = lambda_new

        # Update q in QP problem
        self.qp_problem['q'] = np.append(np.zeros(self.m + self.n),
                                         self.lambda_param * np.ones(self.n))
