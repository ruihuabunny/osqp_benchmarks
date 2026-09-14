import numpy as np
import scipy.sparse as spa
import cvxpy


class EqQPExample(object):
    '''
    Equality constrained QP example
    '''
    def __init__(self, n, seed=1, max_spectrum=None, max_cond_num=None, r=None):
        '''
        Generate problem in QP format and CVXPY format
        n: dimension of the matrix self.P
        seed: random seed; retains the original second positional argument
        r: rank of self.P, from 0 to n; defaults to n
        max_spectrum: the spectrum of self.P should not exceed max_spectrum;
        max_cond_num: the condition number of self.P should not exceed max_cond_num;
        '''
        self.n = int(n)
        self.m = self.n // 2
        self.r = self.n if r is None else r
        if not isinstance(self.r, (int, np.integer)) or not 0 <= self.r <= self.n:
            raise ValueError('r must be an integer between 0 and n')
        rng = np.random.default_rng(seed)

        # The identity block makes G full column rank, including sparse cases.
        lower = spa.random(self.n - self.r, self.r, density=0.15,
                           data_rvs=rng.standard_normal, random_state=rng,
                           format='csc')
        G = spa.vstack([spa.eye(self.r, format='csc'), lower], format='csc')
        G = G[rng.permutation(self.n), :]
        Lambda = spa.diags(rng.uniform(0.1, 1.0, self.r),
                           shape=(self.r, self.r), format='csc')
        self.P = (G @ Lambda @ G.T).tocsc()
        self.A = spa.random(self.m, self.n, density=0.15,
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
