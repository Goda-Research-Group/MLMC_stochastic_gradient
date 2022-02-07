import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Definition for Simple Test Case

eta_test = 0
p0_test = 1 - 2 ** (-1.5)

dist_theta_test = stats.lognorm(s=1, scale=np.exp(0))
sigma_theta_test = 2 * np.log(dist_theta_test.mean()/dist_theta_test.median())
mu_theta_test = np.log(dist_theta_test.median())

dist_epsilon_test = stats.norm(0, 1)
sigma_epsilon_test = dist_epsilon_test.var()


def dist_theta_rvs_test(N, seed):
    return dist_theta_test.rvs(2 * N, random_state=seed).reshape(N, 2)


def dist_theta_pdf_test(theta):
    pdf1 = dist_theta_test.pdf(theta[:, 0])
    pdf2 = dist_theta_test.pdf(theta[:, 1])
    return pdf1 * pdf2


def dist_y_rvs_test(theta, xi, seed):
    log_y = log_y_mu_test(theta, xi)
    epsilon = dist_epsilon_test.rvs(log_y.shape, random_state=seed)
    return np.exp(log_y + epsilon), epsilon


def dist_y_pdf_test(y, theta, xi):
    epsilon = np.log(y) - log_y_mu_test(theta, xi)
    return (dist_epsilon_test.pdf(epsilon) / y).prod(axis=1)


def dist_y_pdf_exponent_test(y, theta, xi, q, importance_sampling_method):
    e_det = sigma_epsilon_test ** len(y)
    exponents = (
        -np.sum(
            (np.log(y) -  log_y_mu_test(theta, xi)) ** 2
            , axis=1
        ) / 2 / sigma_epsilon_test
        - np.sum(np.log(y))
    )
    t_det = sigma_theta_test ** len(theta[0])
    if importance_sampling_method is not None:
        exponents += (
            -np.sum(
                (np.log(theta) -  mu_theta_test) ** 2
                , axis=1
            ) / 2 / sigma_theta_test
            - np.sum(
                np.log(theta)
                , axis=1
            )
        )
        exponents -= (
            -np.sum(
                (theta - q.mean)
                * ((theta - q.mean) @ np.linalg.inv(q.cov)),
                axis=1,
            )
            / 2
        )

    return exponents, e_det, t_det


def g_h_test(xi):
    xi = xi[0]
    g = np.sqrt(np.exp(-xi ** 2))
    h = np.sqrt(1.5 * (1 - g ** 2))
    return np.array([g, h])


def ggrad_test(xi):
    g = g_h_test(xi)[0]
    xi = xi[0]
    return -xi * g


def hgrad_test(xi):
    g = g_h_test(xi)[0]
    h = g_h_test(xi)[1]
    ggrad = ggrad_test(xi)
    return -1.5 * g * ggrad / h


optimal_xi_test = np.sqrt(np.log(3))


def log_y_mu_test(theta, xi):
    return np.log(theta) * g_h_test(xi)


def nabla_log_p_test(y, epsilon, theta, xi):
    return (
        (
            np.log(theta) * np.log(y) / sigma_epsilon_test
            - np.log(theta) ** 2 * g_h_test(xi) / sigma_epsilon_test
        )
        @ np.array([[ggrad_test(xi)], [hgrad_test(xi)]])
    )


def nabla_log_p_reparameterized_test(y, epsilon, theta_outer, theta_inner, xi):
    epsilon = np.log(y) - log_y_mu_test(theta_outer, xi)
    return -(
        (
            np.log(theta_outer)
            + np.log(theta_outer / theta_inner) * epsilon / sigma_epsilon_test
            + np.log(theta_outer / theta_inner) ** 2 * g_h_test(xi) / sigma_epsilon_test
        )
        @ np.array([[ggrad_test(xi)], [hgrad_test(xi)]])
    )


def laplace_approximation_test(theta, y, xi):
    gh_xi = g_h_test(xi)
    nabla_F = (
        - (gh_xi ** 2 / sigma_epsilon_test + 1 / sigma_theta_test)
        * np.log(theta[0])
        + gh_xi / sigma_epsilon_test * np.log(y)
        + mu_theta_test / sigma_theta_test - 1
    ) / theta[0]

    nabla_nabla_F = -np.diag(
        (
            - (gh_xi ** 2 / sigma_epsilon_test + 1 / sigma_theta_test)
            * np.log(theta[0])
            + gh_xi ** 2 / sigma_epsilon_test
            + gh_xi / sigma_epsilon_test * np.log(y)
            + (mu_theta_test + 1) / sigma_theta_test - 1
        ) / (theta[0] ** 2)
    )

    theta_hat = theta[0] - (np.linalg.inv(nabla_nabla_F) @ nabla_F[:, np.newaxis]).squeeze()

    Sigma_hat_I = np.diag(
        (
            - (gh_xi ** 2 / sigma_epsilon_test + 1 / sigma_theta_test)
            * np.log(theta_hat)
            + gh_xi ** 2 / sigma_epsilon_test
            + gh_xi / sigma_epsilon_test * np.log(y)
            + (mu_theta_test + 1) / sigma_theta_test - 1
        ) / (theta_hat ** 2)
    )
    return stats.multivariate_normal(mean=theta_hat, cov=np.linalg.inv(Sigma_hat_I))


model_test = {
    "dist_theta_rvs": dist_theta_rvs_test,
    "dist_theta_pdf": dist_theta_pdf_test,
    "dist_y_rvs": dist_y_rvs_test,
    "dist_y_pdf": dist_y_pdf_test,
    "dist_y_pdf_exponent": dist_y_pdf_exponent_test,
    "nabla_log_p": nabla_log_p_test,
    "nabla_log_p_reparameterized": nabla_log_p_reparameterized_test,
    "eta": eta_test,
    "laplace_approximation": laplace_approximation_test,
    "qY": None,
    "p0": p0_test,
}


def eig_test(xi):
    g_h_xi = g_h_test(xi)
    return (np.log(
        (g_h_xi[0] ** 2 * sigma_theta_test / sigma_epsilon_test + 1)
        * (g_h_xi[1] ** 2 * sigma_theta_test / sigma_epsilon_test + 1)
    )) / 2


def eig_bound_test(xi):
    g_h_xi = g_h_test(xi)
    return (
        g_h_xi[0] ** 2 * sigma_theta_test / sigma_epsilon_test
        + g_h_xi[1] ** 2 * sigma_theta_test / sigma_epsilon_test
    )


# Definition for Pharmacokinetic model

D = 400
eta_pk = 2 ** (2.5)
p0_pk = 0.9

dist_theta_pk = stats.multivariate_normal(
    [np.log(1), np.log(0.1), np.log(20)], np.eye(3) * 0.05
)
dist_epsilon_add_pk = stats.norm(0, np.sqrt(0.1))
sigma_epsilon_add_pk = dist_epsilon_add_pk.var()
dist_epsilon_prop_pk = stats.norm(0, np.sqrt(0.01))
sigma_epsilon_prop_pk = dist_epsilon_prop_pk.var()


def dist_theta_rvs_pk(N, seed):
    return dist_theta_pk.rvs(N, random_state=seed)


def dist_theta_pdf_pk(theta):
    return dist_theta_pk.pdf(theta)


def dist_y_rvs_pk(theta, xi, seed):
    g = g_pk(theta, xi)
    epsilon_add = dist_epsilon_add_pk.rvs(g.shape, random_state=seed)
    epsilon_prop = dist_epsilon_prop_pk.rvs(g.shape, random_state=seed)
    return (
        g + epsilon_prop * g + epsilon_add,
        np.concatenate([epsilon_add, epsilon_prop], 1)
    )


def dist_y_pdf_pk(y, theta, xi):
    g = g_pk(theta, xi)
    sigma_y = sigma_epsilon_add_pk + sigma_epsilon_prop_pk * (g ** 2)
    pdf_array = np.exp(- (y - g) ** 2 / sigma_y /2) / np.sqrt(2 * np.pi * sigma_y)
    return np.prod(pdf_array, axis=1)


def dist_y_pdf_exponent_pk(y, theta, xi, q, importance_sampling_method):
    e_det = 1
    g = g_pk(theta, xi)
    sigma_y = sigma_epsilon_add_pk + sigma_epsilon_prop_pk * (g ** 2)
    exponents = (
        -np.sum(
            (y - g) ** 2 / sigma_y
            , axis=1
        ) / 2
        - np.sum(
            np.log(sigma_y)
            , axis=1
        ) / 2
    )
    t_det = np.linalg.det(dist_theta_pk.cov)
    if importance_sampling_method is not None:
        exponents += (
            -np.sum(
                (theta - dist_theta_pk.mean)
                * (
                    (theta - dist_theta_pk.mean)
                    @ np.linalg.inv(dist_theta_pk.cov)
                ),
                axis=1,
            )
            / 2
        )
        exponents -= (
            -np.sum(
                (theta - q.mean)
                * ((theta - q.mean) @ np.linalg.inv(q.cov)),
                axis=1,
            )
            / 2
        )

    return exponents, e_det, t_det


def g_pk(theta, xi):
    ka = np.exp(theta.T[0])[:, np.newaxis]
    ke = np.exp(theta.T[1])[:, np.newaxis]
    V = np.exp(theta.T[2])[:, np.newaxis]
    xi = xi[np.newaxis, :]
    return (
        D * ka
        * (np.exp(-ke * xi) - np.exp(-ka * xi))
        / (ka - ke) / V
    )


def ggrad_pk(theta, xi):
    ka = np.exp(theta.T[0])[:, np.newaxis]
    ke = np.exp(theta.T[1])[:, np.newaxis]
    V = np.exp(theta.T[2])[:, np.newaxis]
    xi = xi[np.newaxis, :]
    return (
        D * ka
        * (-ke * np.exp(-ke * xi) + ka * np.exp(-ka * xi))
        / (ka - ke) / V
    )[:, :, np.newaxis] * np.eye(15)[np.newaxis, :, :]


def nabla_log_p_pk(y, epsilon, theta, xi):
    g = g_pk(theta, xi)
    sigma_y = sigma_epsilon_add_pk + sigma_epsilon_prop_pk * (g ** 2)
    ggrad = ggrad_pk(theta, xi)
    ggrad_diag = np.array([np.diag(gg) for gg in ggrad])
    return (
        -((1 + sigma_epsilon_prop_pk) * g - y) / sigma_y
        + g * (y - g) ** 2 * sigma_epsilon_prop_pk / (sigma_y ** 2)
    ) * ggrad_diag


def nabla_log_p_reparameterized_pk(y, epsilon, theta_outer, theta_inner, xi):
    epsilon_prop = epsilon[:, 15:]
    g_inner = g_pk(theta_inner, xi)
    sigma_y = sigma_epsilon_add_pk + sigma_epsilon_prop_pk * (g_inner ** 2)
    ggrad_outer = ggrad_pk(theta_outer, xi)
    ggrad_inner = ggrad_pk(theta_inner, xi)
    ggrad_outer_diag = np.array([np.diag(gg) for gg in ggrad_outer])
    ggrad_inner_diag = np.array([np.diag(gg) for gg in ggrad_inner])
    return (
        -(
            ((1 + sigma_epsilon_prop_pk) * g_inner - y) * ggrad_inner_diag
            + (y - g_inner) * (1 + epsilon_prop) * ggrad_outer_diag
        ) / sigma_y
        + g_inner * (y - g_inner) ** 2 * sigma_epsilon_prop_pk
        * ggrad_inner_diag / (sigma_y ** 2)
    )


def J_and_H_pk(theta, xi):
    g = g_pk(theta, xi).squeeze()
    ka = np.exp(theta.T[0])
    ke = np.exp(theta.T[1])
    V = np.exp(theta.T[2])
    term1 = D / V * ka / (ka - ke) * xi * np.exp(-ke * xi)
    term2 = D / V * ka / (ka - ke) * xi * np.exp(-ka * xi)
    term3 = ke / (ka - ke)
    J = np.array(
        [
            term3 * g - ka * term2, - term3 * g + ke * term1, g
        ]
    )
    H = np.array(
        [
            [
                -ka / (ka - ke) * term3 * g - term3 * J[0] - (
                    ka * (ka - 2 * ke) / (ka - ke) - ka ** 2 * xi
                ) * term2,
                term3 * (g + J[0] - J[1]),
                -J[0]
            ],
            [
                term3 * (g + J[0] - J[1]),
                -ka / (ka - ke) * term3 * g + term3 * J[1] + (
                ka * term3 - ke ** 2 * xi
                ) * term1,
                -J[1]
            ],
            [-J[0], -J[1], -g]
        ]
    ).T
    return J, H


def J_pk(theta, xi):
    return J_and_H_pk(theta, xi)[0]


def H_pk(theta, xi):
    return J_and_H_pk(theta, xi)[1]


def dd_log_p_pk(theta, xi):
    return -np.linalg.inv(dist_theta_pk.cov)


def laplace_approximation_pk(theta, y, xi):
    epsilon = y - g_pk(theta, xi).squeeze()
    Sigma_epsilon_I_pk = np.diag(
        1 / (g_pk(theta, xi).squeeze() ** 2 * sigma_epsilon_prop_pk + sigma_epsilon_add_pk)
    )
    J = J_pk(theta, xi)
    H = H_pk(theta, xi)
    theta_hat = (
        theta
        - np.linalg.inv(
            J @ Sigma_epsilon_I_pk @ J.T
            + H[:,:,:,np.newaxis].T @ Sigma_epsilon_I_pk @ epsilon
            - dd_log_p_pk(theta, xi)
        )
        @ J
        @ Sigma_epsilon_I_pk
        @ epsilon
    )
    J_hat = J_pk(theta_hat, xi)
    Sigma_hat = np.linalg.inv(
        J_hat @ Sigma_epsilon_I_pk @ J_hat.T
        - dd_log_p_pk(theta_hat, xi)
    )

    return stats.multivariate_normal(mean=theta_hat[0], cov=Sigma_hat)


model_pk = {
    "dist_theta_rvs": dist_theta_rvs_pk,
    "dist_theta_pdf": dist_theta_pdf_pk,
    "dist_y_rvs": dist_y_rvs_pk,
    "dist_y_pdf": dist_y_pdf_pk,
    "dist_y_pdf_exponent": dist_y_pdf_exponent_pk,
    "nabla_log_p": nabla_log_p_pk,
    "nabla_log_p_reparameterized": nabla_log_p_reparameterized_pk,
    "eta": eta_pk,
    "laplace_approximation": laplace_approximation_pk,
    "qY": None,
    "p0": p0_pk,
}


# Visuzlize xi for Pharmacokinetic model


def show_xi(xi, scale, n_ticks, filename):

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
    plt.plot(
        (np.arange(n_ticks+1)[1:-1]/n_ticks)*(scale[1]-scale[0])+scale[0],
        np.ones(n_ticks-1), marker="|", color="k", markersize=5
    )

    plt.text(scale[0], 0.98, str(int(scale[0])), ha="center", va="top")
    plt.text(scale[1], 0.98, str(int(scale[1])), ha="center", va="top")

    plt.savefig(filename + ".eps")
    print("The graph has been saved at [" + filename + ".eps].")
    plt.close()
