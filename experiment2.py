import os
import numpy as np
import matplotlib.pyplot as plt
from model import OptimalAlgorithm, ThresholdAlgorithm
from data import DataGenerator

def main():
    p_min, p_max = 2, 5
    T = 500
    epsilons = [1e-3, 1e-2, 1e-1, 1]
    data = DataGenerator(p_min , p_max)

    n_experiments = 100

    optimal_algorithm = OptimalAlgorithm()
    threshold_algorithm = ThresholdAlgorithm(p_min, p_max)

    results = {
        "ratio": [],
        "std": [],
        "epsilon": epsilons
    }
    
    for epsilon in epsilons:
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
    plt.plot(results["epsilon"], results["ratio"])
    plt.xlabel("epsilon")
    plt.xscale("log")
    plt.ylabel("Average OPT/ALG")
    plt.savefig(os.path.join(root, "output", "epsilon_vs_T.png"))


if __name__ == "__main__":
    main()