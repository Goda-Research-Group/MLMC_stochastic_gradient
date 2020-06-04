import numpy as np
import matplotlib.pyplot as plt
import mlmc_eig_grad.mlmc_eig as mlmc_eig
import mlmc_eig_grad.models as models


def eig_with_path(paths, labels, n_step, n_sample, filename):
    if len(paths) == 1:
        colors = [0]
    else:
        colors = np.arange(len(paths)) * 0.8 / (len(paths) - 1)

    plt.figure(figsize=(13, 10))

    for i in range(len(paths)):
        path = paths[i]
        times = np.arange(0, len(path) + 1, n_step)
        eig_array = []
        for t in times:
            U = mlmc_eig.randomized_mlmc(
                models.model_pk, n_sample, mlmc_eig.mlmc_eig_value, path[t],
            ).mean(axis=0)
            eig_array.append(U)
        plt.plot(
            times, eig_array, color=str(colors[i]), linewidth=1.3, label=labels[i],
        )

    plt.legend(fontsize=15, loc=(0.85, 0.02))
    plt.xlabel("$iterations$", fontsize=22)
    plt.ylabel("$expected information gain$", fontsize=22)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
    plt.close()
