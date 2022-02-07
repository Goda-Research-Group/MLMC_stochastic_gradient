import numpy as np
import matplotlib.pyplot as plt
import mlmc_eig_grad.mlmc_eig as mlmc_eig
import mlmc_eig_grad.models as models


def eig_mean_with_error(paths_, labels, n_step, eig_func, filename, M0=1, n_sample=20000):
    paths_ = np.array(paths_)
    if len(paths_) == 1:
        colors = [0]
    else:
        colors = np.arange(len(paths_)) * 0.8 / (len(paths_) - 1)

    plt.figure(figsize=(13, 10))

    for i in range(len(paths_)):
        paths = paths_[i]
        times = np.arange(0, paths.shape[1], n_step)
        eig_mean = []
        eig_se = []

        for t in times:
            eig_array = []
            for j in range(len(paths)):
                if eig_func is not None :
                    U = eig_func(paths[j, t])
                else:
                    U = mlmc_eig.randomized_mlmc(
                           models.model_pk, n_sample,
                        mlmc_eig.mlmc_eig_value, paths[j, t], M0
                    ).mean(axis=0)
                eig_array.append(U)
            eig_mean.append(np.mean(eig_array))
            if len(paths) > 1:
                eig_se.append(np.std(eig_array, ddof=1) / np.sqrt(len(paths)))

        eig_mean = np.array(eig_mean)
        plt.plot(
            times, eig_mean, color=str(colors[i]), linewidth=1.3, label=labels[i],
        )
        if len(paths) > 1:
            eig_up = eig_mean + eig_se
            eig_low = eig_mean - eig_se
            plt.plot(
                times, eig_up, color=str(colors[i]), linestyle="dashed", linewidth=1,
            )
            plt.plot(
                times, eig_low, color=str(colors[i]), linestyle="dashed", linewidth=1,
            )
    plt.legend(fontsize=15, loc=(0.85, 0.02))
    plt.xlabel("$iterations$", fontsize=22)
    plt.ylabel("$expected information gain$", fontsize=22)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
    plt.close()
