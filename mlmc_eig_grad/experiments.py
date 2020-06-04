import numpy as np
import mlmc_eig_grad.mlmc_eig as mlmc_eig
import mlmc_eig_grad.optimize as optimize
import mlmc_eig_grad.models as models
import mlmc_eig_grad.visualize_path as visualize_path
import mlmc_eig_grad.visualize_eig as visualize_eig


def main():
    expr_test()
    expr_pk()


# Test Case: experiments in subsection 4.1


def expr_test():

    mlmc_eig.use_importance_sampling = False

    ## convergence of bias/variance (Figure 1)

    num_sample = 20000
    num_level = 10
    xi = [1, 1]
    mlmc_eig.bias_variance_check_and_graph(
        models.model_t,
        mlmc_eig.mlmc_eig_value,
        num_sample,
        num_level,
        xi,
        "eig_value_test",
    )
    mlmc_eig.bias_variance_check_and_graph(
        models.model_t,
        mlmc_eig.mlmc_eig_grad,
        num_sample,
        num_level,
        xi,
        "eig_grad_test",
    )

    ## Optimization (Figure 2)

    optimize.xi_scale = 1
    max_iters = 1000
    batch_size = 100000
    progress_step = 50
    params = {"alpha": 0.004, "beta1": 0.9, "beta2": 0.999}

    ### MLMC

    condition = optimize.get_condition_num_iters(max_iters)
    restriction = optimize.restriction_pass
    gradient = optimize.get_gradient_randomized_mlmc(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size,
    )
    progress = optimize.get_progress_show_list(progress_step)
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    xi_init = [1, 1]

    print("\nMLMC")
    print("steps= 0\nxi=", xi_init)
    path_mlmc = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_mlmc.csv", path_mlmc)

    ### convergence of bias/variance during optimization (Figure 3)

    num_sample = 20000
    num_level = 10
    mlmc_eig.variance_check_with_path(
        models.model_t,
        mlmc_eig.mlmc_eig_grad,
        num_sample,
        num_level,
        path_mlmc,
        "var_with_path_test",
    )

    ### use posterior distribution

    mlmc_eig.use_importance_sampling = True
    mlmc_eig.use_laplace = False
    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 1
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nPosterior")
    print("steps= 0\nxi=", xi_init)
    path_post = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_post.csv", path_post)

    ### stdMC(m=1)

    mlmc_eig.use_importance_sampling = False

    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 1
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=1)")
    print("steps= 0\nxi=", xi_init)
    path_stdmc1 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_stdmc1.csv", path_stdmc1)

    ### stdMC(m=2)

    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 2
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=2)")
    print("steps= 0\nxi=", xi_init)
    path_stdmc2 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_stdmc2.csv", path_stdmc2)

    ### stdMC(m=3)

    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 3
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=3)")
    print("steps= 0\nxi=", xi_init)
    path_stdmc3 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_stdmc3.csv", path_stdmc3)

    ### stdMC(m=10)

    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 10
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=10)")
    print("steps= 0\nxi=", xi_init)
    path_stdmc10 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_stdmc10.csv", path_stdmc10)

    ### stdMC(m=100)

    gradient = optimize.get_gradient_nested(
        models.model_t, mlmc_eig.mlmc_eig_grad, batch_size, 100
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=100)")
    print("steps= 0\nxi=", xi_init)
    path_stdmc100 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("path_stdmc100.csv", path_stdmc100)

    ### visualizing paths

    paths = [
        path_mlmc,
        path_post,
        path_stdmc1,
        path_stdmc2,
        path_stdmc3,
        path_stdmc10,
        path_stdmc100,
    ]
    labels = [
        "MLMC",
        "stdMC(posterior)",
        "stdMC(M=1)",
        "stdMC(M=2)",
        "stdMC(M=3)",
        "stdMC(M=10)",
        "stdMC(M=100)",
    ]
    visualize_path.contour_and_paths(
        models.eig_t, paths, labels, -0.7, 1.8, -1.0, 1.5, "contour_and_paths"
    )
    visualize_path.contour_and_paths(
        models.eig_t, paths, labels, -0.5, 0.2, -0.2, 0.5, "contour_and_paths_zoom"
    )


# Pharmacokinetic model: experiments in subsection 4.2


def expr_pk():

    random_state_xi = np.random.RandomState(1234)

    mlmc_eig.use_importance_sampling = True
    mlmc_eig.use_laplace = True

    ## convergence of bias/variance (Figure 4)

    num_sample = 20000
    num_level = 10
    xi = [15, 30, 45, 60, 90, 120, 180, 360, 480, 720]
    mlmc_eig.bias_variance_check_and_graph(
        models.model_pk,
        mlmc_eig.mlmc_eig_grad,
        num_sample,
        num_level,
        xi,
        "eig_grad_pk",
    )

    ## Optimization

    ### visualization of optimizing process (Figure 5)

    optimize.xi_scale = 240
    max_iters = 10000
    batch_size = 5000
    batch_size_eig = 20000
    progress_step = 2000
    params = {"alpha": 0.24, "beta1": 0.9, "beta2": 0.999}

    condition = optimize.get_condition_num_iters(max_iters)
    restriction = optimize.get_restriction_clip(0, optimize.xi_scale)
    gradient = optimize.get_gradient_randomized_mlmc(
        models.model_pk, mlmc_eig.mlmc_eig_grad, batch_size,
    )
    progress = optimize.get_progress_show_xi(progress_step, batch_size_eig, "run1")

    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }

    print("\nPointSet: ", 1)
    xi_init = random_state_xi.uniform(0, optimize.xi_scale, 10)
    models.show_xi(xi_init, (0, optimize.xi_scale), "run1-0")
    print("steps= 0")
    U_init = mlmc_eig.randomized_mlmc(
        models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init,
    ).mean(axis=0)
    print("EIG=", U_init)
    path1 = optimize.amsgrad(xi_init, functions, params)
    np.savetxt("run1-path.csv", path1)
    print()

    ### convergence of bias/variance during optimization (Figure 6)

    num_sample = 20000
    num_level = 10
    mlmc_eig.variance_check_with_path(
        models.model_pk,
        mlmc_eig.mlmc_eig_grad,
        num_sample,
        num_level,
        path1,
        "var_with_path_pk",
    )

    ### behaviours of EIGs in 5 runs (Figure 7)

    functions["progress"] = optimize.progress_pass

    paths = [path1]

    for i in range(2, 6):
        print("PointSet: ", i)
        xi_init = random_state_xi.uniform(0, optimize.xi_scale, 10)
        print("Begin: ")
        models.show_xi(xi_init, (0, optimize.xi_scale), "run" + str(i) + "-0")
        U_init = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init,
        ).mean(axis=0)
        print("EIG=", U_init)
        path = optimize.amsgrad(xi_init, functions, params)
        paths.append(path)
        print("End: ")
        models.show_xi(
            path[-1], (0, optimize.xi_scale), "run" + str(i) + "-" + str(max_iters)
        )
        U_fin = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, path[-1],
        ).mean(axis=0)
        print("EIG=", U_fin)
        print()
        np.savetxt("run" + str(i) + "-path.csv", path)

    labels = ["run1", "run2", "run3", "run4", "run5"]
    num_step = 100
    num_sample = 20000
    visualize_eig.eig_with_path(paths, labels, num_step, num_sample, "eig_graph")
