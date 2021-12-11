import numpy as np
import cvxpy

class ThresholdAlgorithm:
    def __init__(self, p_min, p_max):
        self.p_min = p_min
        self.p_max = p_max

        self.beta = 1 / (1 + np.log(p_max/p_min))

    def phi(self, y):
        if y < self.beta:
            return self.p_min
        elif self.beta <= y <= 1:
            return self.p_min * np.exp(y/self.beta - 1)
        else:
            return 0

    def solve(self, v, w):
        assert len(v) == len(w)
        T = len(v)
        y_prev = 0
        utility = 0

        for t in range(T):
            value_density = v[t]/w[t]

            if value_density >= self.phi(y_prev) and (y_prev + w[t]) <= 1.0:
                utility += v[t]
                y_prev += w[t]

        return utility

class MultidimensionalThresholdAlgorithm:
    def __init__(self, p_min, p_max):
        self.p_min = p_min
        self.p_max = p_max

        self.beta = 1 / (1 + np.log(p_max/p_min))

    def phi(self, y):
        if y < self.beta:
            return self.p_min
        elif self.beta <= y <= 1:
            return self.p_min * np.exp(y/self.beta - 1)
        else:
            return 0

    def solve(self, v, w):
        assert len(v) == len(w)
        T = len(v)
        y_prev = 0
        utility = 0

        for t in range(T):

            d = len(v[t])
            value_density = v[0]/w[0]

            for j in range(1, d):
                value_density = max(value_density, v[j]/w[j])

            if value_density >= self.phi(y_prev) and (y_prev + w[t]) <= 1.0:
                utility += v[t]
                y_prev += w[t]

        return utility






class OptimalAlgorithm:
    def solve(self, v, w):
        assert len(v) == len(w)

        T = len(v)
        x = cvxpy.Variable(T, boolean=True)

        constraints = cvxpy.multiply(w, x) <= 1
        utility = cvxpy.sum(cvxpy.multiply(v, x))

        problem = cvxpy.Problem(cvxpy.Maximize(utility), [constraints])
        problem.solve(solver='ECOS_BB')

        return problem.value