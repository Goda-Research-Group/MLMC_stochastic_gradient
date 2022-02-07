# Some functions to calculate an EIG and a gradient of EIG.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing as mp
import collections as col
from numpy.random import PCG64, RandomState, SeedSequence

use_multiprocess = True
num_process = mp.cpu_count()
if num_process > 32:
    num_process = 32

seed_sequence = SeedSequence(123456)
random_state_outerloop = RandomState(PCG64(seed_sequence.spawn(1)[0]))
random_state_level_selection = RandomState(PCG64(seed_sequence.spawn(1)[0]))


use_reparametrization = False
importance_sampling_method = None


def mlmc_eig_value_and_grad(model, is_level_0, M, N, xi):

    dist_theta_rvs = model["dist_theta_rvs"]
    dist_theta_pdf = model["dist_theta_pdf"]
    dist_y_rvs = model["dist_y_rvs"]
    dist_y_pdf = model["dist_y_pdf"]
    dist_y_pdf_exponent = model["dist_y_pdf_exponent"]
    nabla_log_p =  model["nabla_log_p"]
    nabla_log_p_reparameterized = model["nabla_log_p_reparameterized"]
    eta = model["eta"]
    laplace_approximation = model["laplace_approximation"]
    qY = model["qY"]

    xi = np.array(xi)

    global mlmc_eig_calc_innerloop

    def mlmc_eig_calc_innerloop(args):
        y, epsilon, theta, seed = args

        random_state_inner = RandomState(PCG64(seed))
        theta = theta[np.newaxis, :]
        epsilon = epsilon[np.newaxis, :]

        if importance_sampling_method is not None:
            if importance_sampling_method == "Laplace":
                q = laplace_approximation(theta, y, xi)
            elif importance_sampling_method == "Posterior":
                q = qY(y, xi)
            theta_inner = q.rvs(size=M, random_state=random_state_inner)
            if theta_inner.ndim <= 1:
                theta_inner = theta_inner[np.newaxis, :]
            p = (
                dist_y_pdf(y, theta_inner, xi)
                * dist_theta_pdf(theta_inner)
                / q.pdf(theta_inner)
            )
        else:
            q = np.nan
            theta_inner = dist_theta_rvs(M, random_state_inner)
            if theta_inner.ndim <= 1:
                theta_inner = theta_inner[np.newaxis, :]
            p = dist_y_pdf(y, theta_inner, xi)

        if np.isscalar(p):
            p = np.array([p])

        nabla_log_p = nabla_log_p_reparameterized(y, epsilon, theta, theta_inner, xi)
        if nabla_log_p.ndim <= 1:
            nabla_log_p = nabla_log_p[np.newaxis, :]

        p_a = p[: int(M / 2)] if not is_level_0 else p
        p_b = p[int(M / 2) :] if not is_level_0 else p
        if(
            is_level_0
            and p.mean() > 0
            or not is_level_0
            and p_a.mean() > 0
            and p_b.mean() > 0
        ):
            log_p_overline = np.log(p.mean())
            log_p_overline_a = (
                np.log(p_a.mean()) if not is_level_0 else np.nan
            )
            log_p_overline_b = (
                np.log(p_b.mean()) if not is_level_0 else np.nan
            )
        else:
            y_dim = len(y)
            exponents, e_det, t_det = dist_y_pdf_exponent(
                y, theta_inner, xi, q, importance_sampling_method
            )

            log_p_overline = logsumexp(exponents,
                importance_sampling_method, q, y_dim, e_det, t_det)
            log_p_overline_a = (
                logsumexp(exponents[: int(M / 2)],
                    importance_sampling_method, q, y_dim, e_det, t_det)
                if not is_level_0
                else np.nan
            )
            log_p_overline_b = (
                logsumexp(exponents[int(M / 2) :],
                    importance_sampling_method, q, y_dim, e_det, t_det)
                if not is_level_0
                else np.nan
            )
            exponents = exponents - exponents.max()
            exponents_a = (
                exponents[: int(M / 2)]
                - exponents[: int(M / 2)].max()
                if not is_level_0
                else exponents
            )
            exponents_b = (
                exponents[int(M / 2) :]
                - exponents[int(M / 2) :].max()
                if not is_level_0
                else exponents
            )
            p = np.exp(exponents)
            p_a = np.exp(exponents_a)
            p_b = np.exp(exponents_b)

        p_overline_rp = (
            p @ nabla_log_p / p.sum()
            if use_reparametrization
            else np.zeros(len(xi))
        )
        p_overline_a_rp = (
            p_a @ nabla_log_p[: int(M / 2)] / p_a.sum()
            if use_reparametrization and not is_level_0
            else p_overline_rp
        )
        p_overline_b_rp = (
            p_b @ nabla_log_p[int(M / 2) :] / p_b.sum()
            if use_reparametrization and not is_level_0
            else p_overline_rp
        )

        return (
            np.repeat(log_p_overline, len(xi)), np.repeat(log_p_overline_a, len(xi)),
            np.repeat(log_p_overline_b, len(xi)), p_overline_rp,
            p_overline_a_rp, p_overline_b_rp
        )

    def logsumexp(r, importance_sampling_method, q, y_dim, e_det, t_det):
        r_max = np.max(r)
        r_ = r - r_max
        log_p_overline = (
            -np.log(len(r_))
            - y_dim / 2 * np.log(2 * np.pi)
            - np.log(e_det) / 2
            + r_max
            + np.log(np.sum(np.exp(r_)))
        )
        if importance_sampling_method is not None:
            log_p_overline += (
                -np.log(t_det) / 2
                + np.log(np.linalg.det(q.cov)) / 2
            )
        return log_p_overline

    theta = dist_theta_rvs(N, random_state_outerloop)
    if theta.ndim <= 1:
        theta = theta[np.newaxis, :]
    y, epsilon = dist_y_rvs(theta, xi, random_state_outerloop)
    if y.ndim <= 1:
        y = y[np.newaxis, :]

    innerloop_args = zip(y, epsilon, theta, seed_sequence.spawn(N))
    if use_multiprocess:
        pool = mp.Pool(num_process)
        (log_p_overline, log_p_overline_a, log_p_overline_b,
        p_overline_rp, p_overline_a_rp, p_overline_b_rp) = np.array(
            pool.map(mlmc_eig_calc_innerloop, innerloop_args)
        ).transpose((1,0,2))
        pool.close()
    else:
        (log_p_overline, log_p_overline_a, log_p_overline_b,
        p_overline_rp, p_overline_a_rp, p_overline_b_rp) = np.array(
            list(map(mlmc_eig_calc_innerloop, innerloop_args))
        ).transpose((1,0,2))

    log_p_overline = log_p_overline[:, 0]
    log_p_overline_a = log_p_overline_a[:, 0]
    log_p_overline_b = log_p_overline_b[:, 0]

    P_l_eig = np.log(dist_y_pdf(y, theta, xi)) - log_p_overline

    if is_level_0:
        Z_l_eig = P_l_eig
    else:
        Z_l_eig = (log_p_overline_a + log_p_overline_b) / 2 - log_p_overline

    if use_reparametrization:
        P_l_eig_grad = (nabla_log_p_reparameterized(y, epsilon, theta, theta, xi)
            - p_overline_rp)
        if is_level_0:
            Z_l_eig_grad = P_l_eig_grad
        else:
            Z_l_eig_grad = (p_overline_a_rp + p_overline_b_rp) / 2 - p_overline_rp
    else:
        nabla_log_p = nabla_log_p(y, epsilon, theta, xi)
        P_l_eig_grad = (P_l_eig - eta)[:, np.newaxis] * nabla_log_p
        if is_level_0:
            Z_l_eig_grad = (Z_l_eig - eta)[:, np.newaxis] * nabla_log_p
        else:
            Z_l_eig_grad = Z_l_eig[:, np.newaxis] * nabla_log_p

    return P_l_eig, Z_l_eig, P_l_eig_grad, Z_l_eig_grad


def mlmc_eig_value(model, is_level_0, M, N, xi):
    return mlmc_eig_value_and_grad(model, is_level_0, M, N, xi)[0:2]


def mlmc_eig_grad(model, is_level_0, M, N, xi):
    return mlmc_eig_value_and_grad(model, is_level_0, M, N, xi)[2:4]


def variance_check_graph(figure, Ps, Zs, title):

    figure.plot(
        np.log2(Ps), marker="^", ms=8, c="0", lw=0.9, ls="--", label="$P_{l}$",
    )
    figure.plot(np.log2(Zs), marker="s", ms=7, c="0", lw=0.9, label="$Z_{l}$")
    figure.legend()
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
    figure.set_xlabel("level $\\ell$")
    figure.set_ylabel("$\\log_2 {\\rm E} \\|\\cdot\\|_2^2$")


def variance_check(model, mlmc_fn, M0, N, L, xi):

    print(
        "log2(E[|P_l|_2])       log2(E[|Z_l|_2])"
    )

    def MLMC_Level_Results(l):
        P, Z = mlmc_fn(model, l == 0, M0 * 2 ** l, N, xi)
        P = P[:, np.newaxis] if P.ndim == 1 else P
        Z = Z[:, np.newaxis] if Z.ndim == 1 else Z
        E2_P = (P ** 2).sum(axis=1).mean()
        E2_Z = (Z ** 2).sum(axis=1).mean()
        print(
            np.log2(E2_P),
            ", ",
            np.log2(E2_Z),
        )
        return E2_P, E2_Z

    return np.array([MLMC_Level_Results(l) for l in range(L + 1)]).T


def variance_check_and_graph(model, mlmc_fn, M0, N, L, xi, filename):

    E2_P_List, E2_Z_List = variance_check(
        model, mlmc_fn, M0, N, L, xi,
    )
    fig, ax = plt.subplots(figsize=(4.5, 5))
    variance_check_graph(ax, E2_P_List, E2_Z_List, "")
    plt.savefig(filename + ".eps", bbox_inches="tight")
    print("The graph has been saved at [" + filename + ".eps].")
    plt.close()


# Implementation for Randomized MLMC


def randomized_mlmc(model, N, mlmc_fn, xi, M0):

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
            mlmc_fn(model, l == 0, M0 * 2 ** l, count, xi)[1] / p_l[l]
            for l, count in col.Counter(levels).items()
        ]
    )


def nested_mc(model, N, M, mlmc_fn, xi):
    P, Z = mlmc_fn(model, True, M, N, xi)
    return np.array(Z)


def variance_check_with_path(model, mlmc_fn, M0, N, L, history, filename):
    xi_list = []
    xi_list.append(history[0])
    xi_list.append(history[int(len(history) / 2)])
    xi_list.append(history[len(history) - 1])
    labels = ["t=0", "t=T/2", "t=T"]

    fig, subps = plt.subplots(ncols=3, figsize=(15, 5))

    P_list = []
    Z_list = []
    for xi, label, subp in zip(xi_list, labels, subps):
        print("Variance checking for [" + label + "]")
        Ps, Zs = variance_check(model, mlmc_fn, M0, N, L, xi)
        variance_check_graph(subp, Ps, Zs, label)

    plt.savefig(filename + ".eps")
    print("The graphs has been saved at [" + filename + ".eps].")
    plt.close()
