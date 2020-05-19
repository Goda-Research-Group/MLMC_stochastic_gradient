import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Definition for Simple Test Case described in subsection 4.1

eta_t = 2.5
p0_t = 1 - 2 ** (-1.5)

dist_theta_t = stats.multivariate_normal([1, 0], [[0.01, -0.0025], [-0.0025, 0.01]])
dist_epsilon_t = stats.multivariate_normal(
    [0, 0, 0], [[0.02, -0.005, 0], [-0.005, 0.02, -0.005], [0, -0.005, 0.02]]
)


def A(xi):
    return np.array(
        [
            [np.exp(-((xi[0] - xi[1]) ** 2)), 2 * np.exp(-xi[1] ** 2)],
            [2 * np.exp(-xi[1] ** 2), 3 * np.sin(xi[0])],
            [3 * np.exp(-xi[0] ** 2), 4 * np.exp(-((xi[0] + xi[1]) ** 2))],
        ]
    )


def dA(xi):
    return np.array(
        [
            [
                [np.exp(-((xi[0] - xi[1]) ** 2)) * (-2 * (xi[0] - xi[1])), 0],
                [0, 3 * np.cos(xi[0])],
                [
                    3 * np.exp(-xi[0] ** 2) * (-2 * xi[0]),
                    4 * np.exp(-((xi[0] + xi[1]) ** 2)) * (-2 * (xi[0] + xi[1])),
                ],
            ],
            [
                [
                    np.exp(-((xi[0] - xi[1]) ** 2)) * (2 * (xi[0] - xi[1])),
                    2 * np.exp(-xi[1] ** 2) * (-2 * xi[1]),
                ],
                [2 * np.exp(-xi[1] ** 2) * (-2 * xi[1]), 0],
                [0, 4 * np.exp(-((xi[0] + xi[1]) ** 2)) * (-2 * (xi[0] + xi[1])),],
            ],
        ]
    ).T


def g_t(theta, xi):
    return (A(xi) @ theta[:, :, np.newaxis]).squeeze(axis=2)


def ggrad_t(theta, xi):
    return (dA(xi)[np.newaxis, :, :, :] * theta[:, :, np.newaxis, np.newaxis]).sum(
        axis=1
    )


def J_t(theta, xi):
    return -A(xi).T


def H_t(theta, xi):
    return np.array([[[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]]).T


def dd_log_p_t(theta, xi):
    return -np.linalg.inv(dist_theta_t.cov)


def qY_t(y, xi):
    Sigma_epsilon_I = np.linalg.inv(dist_epsilon_t.cov)
    Sigma_theta_I = np.linalg.inv(dist_theta_t.cov)

    cov_post = np.linalg.inv(A(xi).T @ Sigma_epsilon_I @ A(xi) + Sigma_theta_I)
    mean_post = cov_post.dot(
        A(xi).T @ Sigma_epsilon_I @ y + Sigma_theta_I @ dist_theta_t.mean
    )
    return stats.multivariate_normal(mean=mean_post, cov=cov_post)


model_t = {
    "dist_epsilon": dist_epsilon_t,
    "dist_theta": dist_theta_t,
    "g": g_t,
    "ggrad": ggrad_t,
    "eta": eta_t,
    "J": J_t,
    "H": H_t,
    "dd_log_p": dd_log_p_t,
    "qY": qY_t,
    "p0": p0_t,
}


def eig_t(xi):

    Sigma_theta = dist_theta_t.cov
    Sigma_epsilon_I = np.linalg.inv(dist_epsilon_t.cov)

    return np.log(
        np.linalg.det(Sigma_epsilon_I @ A(xi) @ Sigma_theta @ A(xi).T + np.eye(3)) / 2
    )


# Definition for Pharmacokinetic model described in subsection 4.2

D = 500
Tmin = 30
eta_pk = 2 ** (2.5)
p0_pk = 0.9

dist_theta_pk = stats.multivariate_normal(
    [-3.26, 8.99], [[0.0071, -0.0057], [-0.0057, 0.0080]]
)
dist_epsilon_pk = stats.multivariate_normal(np.zeros(10), np.eye(10) * (10 ** (-8)))


def g_pk(theta, xi):
    ke = np.exp(theta.T[0])[:, np.newaxis]
    V = np.exp(theta.T[1])[:, np.newaxis]
    xi = xi[np.newaxis, :]
    return (
        D
        * (np.exp(-ke * np.maximum(xi - Tmin, 0)) - np.exp(-ke * xi))
        / (Tmin * ke * V)
    )


def ggrad_pk(theta, xi):
    ke = np.exp(theta.T[0])[:, np.newaxis, np.newaxis]
    V = np.exp(theta.T[1])[:, np.newaxis, np.newaxis]
    xi = xi[np.newaxis, :, np.newaxis]
    indct = np.where(xi - Tmin > 0, 1, 0)
    return (
        D
        * (np.exp(-ke * xi) - indct * np.exp(-ke * np.maximum(xi - Tmin, 0)))
        / (Tmin * V)
    ) * np.eye(10)[np.newaxis, :, :]


def J_and_H_pk(theta, xi):
    g_value = g_pk(theta, xi).squeeze(axis=0)
    ke = np.exp(theta.T[0])
    V = np.exp(theta.T[1])
    term1 = np.exp(-ke * np.maximum(xi - Tmin, 0))
    term2 = np.exp(-ke * xi)
    log_term1 = ke * np.maximum(xi - Tmin, 0)
    log_term2 = ke * xi
    J = np.array(
        [
            g_value + D * (log_term1 * term1 - log_term2 * term2) / (Tmin * ke * V),
            g_value,
        ]
    )
    H = np.array(
        [
            [
                [
                    -J[0]
                    + D
                    * (-(log_term1 ** 2) * term1 + log_term2 ** 2 * term2)
                    / (Tmin * ke * V),
                    -J[0],
                ],
                [-J[0], -g_value],
            ]
        ]
    ).T
    return J, H


def J_pk(theta, xi):
    return J_and_H_pk(theta, xi)[0]


def H_pk(theta, xi):
    return J_and_H_pk(theta, xi)[1]


def dd_log_p_pk(theta, xi):
    return -np.linalg.inv(dist_theta_pk.cov)


model_pk = {
    "dist_epsilon": dist_epsilon_pk,
    "dist_theta": dist_theta_pk,
    "g": g_pk,
    "ggrad": ggrad_pk,
    "eta": eta_pk,
    "J": J_pk,
    "H": H_pk,
    "dd_log_p": dd_log_p_pk,
    "qY": None,
    "p0": p0_pk,
}


# Visuzlize xi for Pharmacokinetic model (for Figure 5, 8, 9)


def show_xi(xi, scale, filename):

    xi = np.maximum(scale[0], xi)
    xi = np.minimum(scale[1], xi)

    fig, _ = plt.subplots(figsize=(15, 15))
    fig.set_figheight(1)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    plt.plot(xi, np.ones_like(xi), marker="o", color="k", markersize=5)
    plt.plot(scale, np.ones(2), marker="|", color="k", markersize=12)

    plt.text(scale[0], 0.98, "0", ha="center", va="top")
    plt.text(scale[1], 0.98, "240", ha="center", va="top")

    plt.savefig(filename + ".eps")
    print("The graph has been saved at [" + filename + ".eps].")
