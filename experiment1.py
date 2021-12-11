import os
import numpy as np
import matplotlib.pyplot as plt
from model import OptimalAlgorithm, ThresholdAlgorithm
from data import DataGenerator

def main():
    p_min, p_max = 2, 5
    epsilon = 1e-2
    data = DataGenerator(p_min , p_max)

    n_experiments = 100
    Ts = [100, 1000, 10000, 10000]

    optimal_algorithm = OptimalAlgorithm()
    threshold_algorithm = ThresholdAlgorithm(p_min, p_max)

    results = {
        "ratio": [],
        "std": [],
        "T": Ts
    }
    
    for T in Ts:
        ratios = []
        for _ in range(n_experiments):
            v, w = data.generate(T, epsilon=epsilon)

            OPT = optimal_algorithm.solve(v, w)
            ALG = threshold_algorithm.solve(v, w)
            ratio = OPT/ALG
            
            ratios.append(ratio)

        ratios = np.array(ratios)

        results["ratio"].append(ratios.mean())
        results["std"].append(ratios.std())


    root = os.path.dirname(__file__)
    plt.plot(results["T"], results["ratio"])
    plt.xlabel("T")
    plt.ylabel("Average OPT/ALG")
    plt.savefig(os.path.join(root, "output", "ratio_vs_T.png"))


if __name__ == "__main__":
    main()