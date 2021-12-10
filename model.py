import numpy as np

class ThesholdAlgorithm:
    def __init__(self, p_min, p_max):
        self.p_min = p_min
        self.p_max = p_max

        self.beta = 1 / (1 + np.log(p_max/p_min))

    def phi(self, y):
        if y < self.beta:
            return self.p_min
        else:
            return self.p_min * np.exp(y/self.beta - 1)

    def solve(self, v, w):
        assert len(v) == len(w)
        T = len(v)
        y_prev = 0
        utility = 0

        for t in range(T):
            value_density = v[t]/w[t]

            if value_density >= self.phi(y_prev):
                utility += v[t]
                y_prev += w[t]

        return utility




class OptimalAlgorithm:
    def __init__(self):
        pass

    def solve(self, v, w):
        pass    