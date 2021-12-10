import numpy as np

class DataGenerator:
    def __init__(self, p_min, p_max):
        self.p_min = p_min
        self.p_max = p_max

    def generate(self, T, epsilon=1e-8):
        w = np.random.uniform(0, epsilon)
        v = np.random.uniform(self.p_min, self.p_max) * w

        return v, w
