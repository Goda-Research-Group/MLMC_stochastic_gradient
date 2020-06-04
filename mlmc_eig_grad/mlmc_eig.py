# Some functions to calculate an EIG and a gradient of EIG.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing as mp
import collections as col
from numpy.random import PCG64, RandomState, SeedSequence

use_multiprocess = True
num_process = mp.cpu_count()
if num_process > 8:
    num_process = 8

seed_sequence = SeedSequence(123456)
random_state_outerloop = RandomState(PCG64(seed_sequence.spawn(1)[0]))
random_state_level_selection = RandomState(PCG64(seed_sequence.spawn(1)[0]))

use_importance_sampling = False
use_laplace = True


def mlmc_eig_value_and_grad(model, is_level_0, M, N, xi):

    global dist_epsilon, dist_theta, g, ggrad, eta, J, H, dd_log_p, qY
    dist_epsilon = model["dist_epsilon"]
    dist_theta = model["dist_theta"]
    g = model["g"]
    ggrad = model["ggrad"]
    eta = model["eta"]
    J = model["J"]
    H = model["H"]
    dd_log_p = model["dd_log_p"]
    qY = model["qY"]

    xi = np.array(xi)

    Sigma_epsilon_I = np.linalg.inv(dist_epsilon.cov)

    global mlmc_eig_calc_innerloop

    def mlmc_eig_calc_innerloop(args):
        y, theta, M, seed, is_level_0 = args

        random_state_inner = RandomState(PCG64(seed))
        theta = theta[np.newaxis, :]
        if use_importance_sampling:
            if use_laplace:
                q = mlmc_eig_laplace_approximation(
                    theta, y - g(theta, xi).squeeze(axis=0), xi
                )
            else:
                q = qY(y, xi)
            theta_inner = q.rvs(size=M, random_state=random_state_inner)
            if M <= 1:
                theta_inner = theta_inner[np.newaxis, :]
            p = (
                dist_epsilon.pdf(y - g(theta_inner, xi))
                * dist_theta.pdf(theta_inner)
                / q.pdf(theta_inner)
            )
        else:
            q = np.nan
            theta_inner = dist_theta.rvs(size=M, random_state=random_state_inner)
            if M <= 1:
                theta_inner = theta_inner[np.newaxis, :]
            p = dist_epsilon.pdf(y - g(theta_inner, xi))

        if (
            is_level_0
            and p.mean() > 0
            or not is_level_0
            and p[: int(M / 2)].mean() > 0
            and p[int(M / 2) :].mean() > 0
        ):
            log_p_overline = np.log(p.mean())
            log_p_overline_a = (
                np.log(p[: int(M / 2)].mean()) if not is_level_0 else np.nan
            )
            log_p_overline_b = (
                np.log(p[int(M / 2) :].mean()) if not is_level_0 else np.nan
            )
        else:
            e_det = np.linalg.det(dist_epsilon.cov)
            y_dim = len(y)
            expornents = (
                -np.sum(
                    (y - g(theta_inner, xi))
                    * ((y - g(theta_inner, xi)) @ Sigma_epsilon_I),
                    axis=1,
                )
                / 2
            )
            if use_importance_sampling:
                expornents += (
                    -np.sum(
                        (theta_inner - dist_theta.mean)
                        * (
                            (theta_inner - dist_theta.mean)
                            @ np.linalg.inv(dist_theta.cov)
                        ),
                        axis=1,
                    )
                    / 2
                )
                expornents -= (
                    -np.sum(
                        (theta_inner - q.mean)
                        * ((theta_inner - q.mean) @ np.linalg.inv(q.cov)),
                        axis=1,
                    )
                    / 2
                )
            log_p_overline = logsumexp(expornents, use_importance_sampling, q)
            log_p_overline_a = (
                logsumexp(expornents[: int(M / 2)], use_importance_sampling, q)
                if not is_level_0
                else np.nan
            )
            log_p_overline_b = (
                logsumexp(expornents[int(M / 2) :], use_importance_sampling, q)
                if not is_level_0
                else np.nan
            )

        return (log_p_overline, log_p_overline_a, log_p_overline_b)

    global logsumexp

    def logsumexp(r, use_importance_sampling, q):
        r_max = np.max(r)
        r_ = r - r_max
        log_p_overline = (
            -np.log(len(r_))
            - y_dim / 2 * np.log(2 * np.pi)
            - np.log(e_det) / 2
            + r_max
            + np.log(np.sum(np.exp(r_)))
        )
        if use_importance_sampling:
            log_p_overline += (
                -np.log(np.linalg.det(dist_theta.cov)) / 2
                + np.log(np.linalg.det(q.cov)) / 2
            )
        return log_p_overline

    global mlmc_eig_laplace_approximation

    def mlmc_eig_laplace_approximation(theta, epsilon, xi):

        theta_hat = (
            theta
            - np.linalg.inv(
                J(theta, xi) @ Sigma_epsilon_I @ J(theta, xi).T
                + H(theta, xi).T @ Sigma_epsilon_I @ epsilon
                - dd_log_p(theta, xi)
            )
            @ J(theta, xi)
            @ Sigma_epsilon_I
            @ epsilon
        )
        Sigma_hat = np.linalg.inv(
            J(theta_hat, xi) @ Sigma_epsilon_I @ J(theta_hat, xi).T
            - dd_log_p(theta_hat, xi)
        )

        return stats.multivariate_normal(mean=theta_hat[0], cov=Sigma_hat)

    theta = dist_theta.rvs(size=N, random_state=random_state_outerloop)
    if theta.ndim <= 1:
        theta = theta[np.newaxis, :]
    epsilon = dist_epsilon.rvs(size=N, random_state=random_state_outerloop)
    if epsilon.ndim <= 1:
        epsilon = epsilon[np.newaxis, :]

    y = g(theta, xi) + epsilon
    dy = ggrad(theta, xi)

    innerloop_args = zip(
        y, theta, np.repeat(M, N), seed_sequence.spawn(N), np.repeat(is_level_0, N)
    )
    if use_multiprocess:
        pool = mp.Pool(num_process)
        log_p_overline, log_p_overline_a, log_p_overline_b = np.array(
            pool.map(mlmc_eig_calc_innerloop, innerloop_args)
        ).T
        pool.close()
    else:
        log_p_overline, log_p_overline_a, log_p_overline_b = np.array(
            list(map(mlmc_eig_calc_innerloop, innerloop_args))
        ).T

    nabla_log_p = (dy * ((Sigma_epsilon_I @ epsilon[:, :, np.newaxis]))).sum(axis=1)

    P_l_eig = np.log(dist_epsilon.pdf(epsilon)) - log_p_overline
    P_l_eig_grad = (P_l_eig - eta)[:, np.newaxis] * nabla_log_p

    if is_level_0:
        Z_l_eig = P_l_eig
        Z_l_eig_grad = (Z_l_eig - eta)[:, np.newaxis] * nabla_log_p
    else:
        Z_l_eig = (log_p_overline_a + log_p_overline_b) / 2 - log_p_overline
        Z_l_eig_grad = Z_l_eig[:, np.newaxis] * nabla_log_p

    return P_l_eig, Z_l_eig, P_l_eig_grad, Z_l_eig_grad


def mlmc_eig_value(model, is_level_0, M, N, xi):
    return mlmc_eig_value_and_grad(model, is_level_0, M, N, xi)[0:2]


def mlmc_eig_grad(model, is_level_0, M, N, xi):
    return mlmc_eig_value_and_grad(model, is_level_0, M, N, xi)[2:4]


def variance_check_graph(isVariance, figure, Ps, Zs, title):

    figure.plot(
        np.log2(Ps), marker="^", ms=8, c="0", lw=0.9, ls="--", label="$P_{l}$",
    )
    figure.plot(np.log2(Zs), marker="s", ms=7, c="0", lw=0.9, label="$Z_{l}$")
    figure.legend()
    if isVariance:
        L = len(Ps)
        beta = (
            (L / 2 * np.log2(Zs)[1:].sum() - np.log2(Zs)[1:].dot(np.arange(1, L)))
            * 12
            / L
            / (L - 1)
            / (L - 2)
        )
        figure.text(0, np.log2(Zs[-5]), "$\\beta$=" + str(round(beta, 3)))
    figure.set_title(title)


def bias_variance_check(model, mlmc_fn, N, L, xi):

    print(
        "E[|P_l|_1]             E[|Z_l|_1]             E[|P_l|_2]             E[|Z_l|_2]"
    )

    def MLMC_Level_Results(l):
        P, Z = mlmc_fn(model, l == 0, 2 ** l, N, xi)
        P = P[:, np.newaxis] if P.ndim == 1 else P
        Z = Z[:, np.newaxis] if Z.ndim == 1 else Z
        E1_P, E2_P = abs(P).sum(axis=1).mean(), (P ** 2).sum(axis=1).mean()
        E1_Z, E2_Z = abs(Z).sum(axis=1).mean(), (Z ** 2).sum(axis=1).mean()
        print(
            np.log2(np.abs(E1_P)),
            ", ",
            np.log2(np.abs(E1_Z)),
            ", ",
            np.log2(E2_P),
            ", ",
            np.log2(E2_Z),
        )
        return E1_P, E1_Z, E2_P, E2_Z

    return np.array([MLMC_Level_Results(l) for l in range(L + 1)]).T


def bias_variance_check_and_graph(model, mlmc_fn, N, L, xi, filename):

    E1_P_List, E1_Z_List, E2_P_List, E2_Z_List = bias_variance_check(
        model, mlmc_fn, N, L, xi
    )
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    variance_check_graph(False, ax1, E1_P_List, E1_Z_List, "E1")
    variance_check_graph(True, ax2, E2_P_List, E2_Z_List, "E2")
    plt.savefig(filename + ".eps")
    print("The graph has been saved at [" + filename + ".eps].")
    plt.close()


# Implementation for Randomized MLMC


def randomized_mlmc(model, N, mlmc_fn, xi):

    p = 2 ** (-1.5)
    p0 = model["p0"]

    def level():
        l = 0
        x = stats.uniform.rvs(random_state=random_state_level_selection)
        if x < p0:
            return l
        else:
            l += 1

        while (x - p0) / (1 - p0) < p ** l:
            l += 1
        return l

    levels = [level() for i in range(N)]

    p_l = [p0] + [(1 - p0) * (1 - p) * p ** l for l in range(100)]

    return np.concatenate(
        [
            mlmc_fn(model, l == 0, 2 ** l, count, xi)[1] / p_l[l]
            for l, count in col.Counter(levels).items()
        ]
    )


def nested_mc(model, N, M, mlmc_fn, xi):
    P, Z = mlmc_fn(model, True, M, N, xi)
    return np.array(Z)


def variance_check_with_path(model, mlmc_fn, N, L, history, filename):
    xi_list = []
    xi_list.append(history[0])
    xi_list.append(history[int(len(history) / 2)])
    xi_list.append(history[len(history) - 1])
    labels = ["t=0", "t=T/2", "t=T"]

    fig, subps = plt.subplots(ncols=3, figsize=(14, 5))

    for xi, label, subp in zip(xi_list, labels, subps):
        print("Variance checking for [" + label + "]")
        _, _, Ps, Zs = bias_variance_check(model, mlmc_fn, N, L, xi)
        variance_check_graph(True, subp, Ps, Zs, label)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
    plt.close()
