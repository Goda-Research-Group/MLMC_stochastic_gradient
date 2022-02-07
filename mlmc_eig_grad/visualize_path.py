import numpy as np
import matplotlib.pyplot as plt
import mlmc_eig_grad.models as models


def convergence_with_path(paths_, inner_samples, labels, filename):

    if len(paths_) == 1:
        colors = [0]
    else:
        colors = np.arange(len(paths_)) * 0.8 / (len(paths_) - 1)

    plt.figure(figsize=(13, 10))
    plt.xscale('log')
    plt.yscale('log')

    for i in range(len(paths_)):
        paths = np.array(paths_[i]).squeeze()
        times = np.arange(1, paths.shape[1])
        distance_mse = np.mean((paths - models.optimal_xi_test) ** 2, axis=0)
        plt.plot(
            times * inner_samples[i],
            distance_mse[1:],
            color=str(colors[i]),
            linewidth=1.3,
            label=labels[i],
        )
    plt.legend(fontsize=15)
    plt.xlabel("$model evaluation$", fontsize=22)
    plt.ylabel("$mse$", fontsize=22)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
    plt.close()
