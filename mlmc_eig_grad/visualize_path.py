import numpy as np
import matplotlib.pyplot as plt


def contour_and_paths(eig_func, paths, labels, left, right, bottom, top, filename):

    x = np.linspace(left, right, 200)
    y = np.linspace(bottom, top, 200)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            Z[i, j] = eig_func([X[i, j], Y[i, j]])

    plt.figure(figsize=(13, 10))
    plt.contour(x, y, Z, levels=20, colors="0.9")

    paths = np.array(paths)
    num_paths = len(paths)
    linestyles = ["--"] + ["-"] * (num_paths - 1)
    if num_paths == 1:
        colors = ["0"]
    else:
        colors = np.arange(num_paths) * 0.8 / (num_paths - 1)
    markers = ["*", "o", "^", "v", "s", "D", "p", "h"]
    m_sizes = [19, 14, 15, 15, 13, 13, 16, 16]

    for i in range(num_paths):
        plt.plot(
            paths[i, :, 0],
            paths[i, :, 1],
            color=str(colors[i]),
            ls=linestyles[i],
            linewidth=1.3,
        )
        plt.plot(
            paths[i, -1, 0],
            paths[i, -1, 1],
            color="1",
            marker=markers[i],
            ms=m_sizes[i],
            markeredgewidth=2.5,
            markeredgecolor="0",
            label=labels[i],
        )

    plt.legend(fontsize=15)
    plt.xlabel("$\\xi_1$", fontsize=23)
    plt.ylabel("$\\xi_2$", fontsize=23)
    plt.xlim(left, right)
    plt.ylim(bottom, top)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
