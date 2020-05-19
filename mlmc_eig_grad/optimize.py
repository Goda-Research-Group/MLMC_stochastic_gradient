import numpy as np
import mlmc_eig_grad.mlmc_eig as mlmc_eig
import mlmc_eig_grad.models as models

xi_scale = 1


def amsgrad_initialize(xi):
    return np.zeros(len(xi)), np.zeros(len(xi)), np.zeros(len(xi))


def amsgrad_iterate(xi_prev, gradient, states, params):
    m, v, v_ = states
    alpha = params["alpha"] / xi_scale
    beta1 = params["beta1"]
    beta2 = params["beta2"]

    g_t = xi_scale * gradient(xi_prev * xi_scale)

    m = beta1 * m + (1 - beta1) * g_t
    v = beta2 * v + (1 - beta2) * g_t ** 2
    v_ = np.maximum(v, v_)
    xi = xi_prev + alpha * m / np.sqrt(v_)
    return xi, (m, v, v_)


def stochastic_gradient(initialize, iterate, xi_init, functions, params):
    condition = functions["condition"]
    restriction = functions["restriction"]
    gradient = functions["gradient"]
    progress = functions["progress"]

    xi_prev = np.array(xi_init) / xi_scale
    history = [xi_init]
    states = initialize(xi_prev)
    while condition(history):
        xi, states = iterate(xi_prev, gradient, states, params)
        xi = restriction(xi * xi_scale) / xi_scale
        history.append(xi * xi_scale)
        progress(history)
        xi_prev = xi
    return np.array(history)


def amsgrad(xi_init, functions, params):
    return stochastic_gradient(
        amsgrad_initialize, amsgrad_iterate, xi_init, functions, params
    )


def get_condition_num_iters(max_iters):
    global max_iters_g
    max_iters_g = max_iters
    return condition_num_iters


def condition_num_iters(history):
    return len(history) - 1 < max_iters_g


def get_restriction_clip(xi_min, xi_max):
    global xi_min_g, xi_max_g
    xi_min_g = xi_min
    xi_max_g = xi_max
    return restriction_clip


def restriction_clip(xi):
    xi = np.maximum(xi_min_g, xi)
    xi = np.minimum(xi_max_g, xi)
    return xi


def restriction_pass(xi):
    return xi


def get_gradient_randomized_mlmc(model, mlmc_func, batch_size):
    global model_g, mlmc_func_g, batch_size_g
    model_g = model
    mlmc_func_g = mlmc_func
    batch_size_g = batch_size
    return gradient_randomized_mlmc


def get_gradient_nested(model, nmc_func, batch_size, M):
    global model_g, nmc_func_g, batch_size_g, M_g
    model_g = model
    nmc_func_g = nmc_func
    batch_size_g = batch_size
    M_g = M
    return gradient_nested_mc


def gradient_randomized_mlmc(xi):
    return mlmc_eig.randomized_mlmc(model_g, batch_size_g, mlmc_func_g, xi).mean(axis=0)


def gradient_nested_mc(xi):
    return mlmc_eig.nested_mc(model_g, batch_size_g, M_g, nmc_func_g, xi).mean(axis=0)


def get_progress_show_xi(progress_step, batch_size_eig, filename):
    global progress_step_g, batch_size_eig_g, filename_g
    progress_step_g = progress_step
    batch_size_eig_g = batch_size_eig
    filename_g = filename
    return progress_show_xi


def progress_show_xi(history):
    length = len(history)
    if (length - 1) % progress_step_g == 0:
        print("steps=", length - 1)
        U = mlmc_eig.randomized_mlmc(
            models.model_pk,
            batch_size_eig_g,
            mlmc_eig.mlmc_eig_value,
            history[len(history) - 1],
        ).mean(axis=0)
        print("EIG=", U)
        models.show_xi(
            history[-1], (0, xi_scale), filename_g + "-" + str(length - 1),
        )


def get_progress_show_list(progress_step):
    global progress_step_g
    progress_step_g = progress_step
    return progress_show_list


def progress_show_list(history):
    length = len(history)
    if (length - 1) % progress_step_g == 0:
        print("steps=", length - 1)
        print("xi=", history[-1])


def progress_pass(history):
    pass
