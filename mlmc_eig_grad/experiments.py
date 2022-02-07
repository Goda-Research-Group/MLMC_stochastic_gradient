import numpy as np
import matplotlib.pyplot as plt
import mlmc_eig_grad.mlmc_eig as mlmc_eig
import mlmc_eig_grad.optimize as optimize
import mlmc_eig_grad.models as models
import mlmc_eig_grad.visualize_path as visualize_path
import mlmc_eig_grad.visualize_eig as visualize_eig


def main():
    expr_test()
    expr_pk()


def expr_test():

    mlmc_eig.use_reparametrization = True
    mlmc_eig.importance_sampling_method = None

    # The expected information gain and its upper bound (Figure 1)

    xi_array = np.linspace(0, 3, 100)
    U = [models.eig_test([xi]) for xi in xi_array]
    U_bound = [models.eig_bound_test([xi]) for xi in xi_array]
    plt.figure(figsize=(7, 7))
    plt.plot(xi_array, U, color="0", label="U")
    plt.plot(xi_array, U_bound, color="0", linestyle="dashed", label="$\\~{U}$")
    plt.xlim(0, 3)
    plt.ylim(0.2, 1.6)
    plt.legend(fontsize=15)
    plt.xlabel("$\\xi$", fontsize=15)
    plt.ylabel("$U, \\~{U}$", fontsize=15)
    plt.savefig("eig_and_bound_test.eps")
    plt.close()

    ## convergence of variance (Fgure 2)

    M0 = 1
    num_sample = 100000
    num_level = 10
    xi = [1.5]
    mlmc_eig.variance_check_and_graph(
        models.model_test,
        mlmc_eig.mlmc_eig_grad,
        M0,
        num_sample,
        num_level,
        xi,
        "eig_grad_test1",
    )

    xi = [np.sqrt(np.log(3))]
    mlmc_eig.variance_check_and_graph(
        models.model_test,
        mlmc_eig.mlmc_eig_grad,
        M0,
        num_sample,
        num_level,
        xi,
        "eig_grad_test2",
    )

    ## Optimization

    optimize.xi_scale = 1
    max_iters = np.floor(10000000 / 2.21)
    batch_size = 2000
    M0 = 1
    progress_step = 800000
    params = {"alpha": 5}

    ### MLMC

    condition = optimize.get_condition_num_iters(max_iters)
    restriction = optimize.restriction_pass
    gradient = optimize.get_gradient_randomized_mlmc(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, M0
    )
    progress = optimize.get_progress_show_list(progress_step)
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    xi_init = [1.5]

    print("\nMLMC")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_mlmc = optimize.robbins_monro(xi_init, functions, params)
    path_mlmc = optimize.Polyak_Ruppert_averaging(path_mlmc)
    print()

    paths_mlmc = [path_mlmc]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_mlmc.append(path)
        print()

    ### stdMC(m=1)

    max_iters = 10000000
    progress_step = 2000000
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 1
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=1)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc1 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc1 = optimize.Polyak_Ruppert_averaging(path_stdmc1)
    print()

    paths_stdmc1 = [path_stdmc1]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc1.append(path)
        print()

    ### stdMC(m=2)

    max_iters = 5000000
    progress_step = 1000000
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 2
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=2)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc2 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc2 = optimize.Polyak_Ruppert_averaging(path_stdmc2)
    print()

    paths_stdmc2 = [path_stdmc2]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc2.append(path)
        print()

    ### stdMC(m=4)

    max_iters = 2500000
    progress_step = 500000
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 4
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=4)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc4 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc4 = optimize.Polyak_Ruppert_averaging(path_stdmc4)
    print()

    paths_stdmc4 = [path_stdmc4]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc4.append(path)
        print()

    ### stdMC(m=8)

    max_iters = 1250000
    progress_step = 250000
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 8
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=8)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc8 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc8 = optimize.Polyak_Ruppert_averaging(path_stdmc8)
    print()

    paths_stdmc8 = [path_stdmc8]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc8.append(path)
        print()

    ### stdMC(m=16)

    max_iters = 625000
    progress_step = 125000
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 16
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=16)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc16 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc16 = optimize.Polyak_Ruppert_averaging(path_stdmc16)
    print()

    paths_stdmc16 = [path_stdmc16]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc16.append(path)
        print()


    ### stdMC(m=32)

    max_iters = 312500
    progress_step = 62500
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 32
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=32)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc32 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc32 = optimize.Polyak_Ruppert_averaging(path_stdmc32)
    print()

    paths_stdmc32 = [path_stdmc32]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc32.append(path)
        print()


    ### stdMC(m=64)

    max_iters = 156250
    progress_step = 31250
    condition = optimize.get_condition_num_iters(max_iters)
    progress = optimize.get_progress_show_list(progress_step)
    gradient = optimize.get_gradient_nested(
        models.model_test, mlmc_eig.mlmc_eig_grad, batch_size, 64
    )
    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }
    print("\nstdMC(M=64)")
    print("PointSet: 1")
    print("steps= 0\nxi=", xi_init)
    path_stdmc64 = optimize.robbins_monro(xi_init, functions, params)
    path_stdmc64 = optimize.Polyak_Ruppert_averaging(path_stdmc64)
    print()

    paths_stdmc64 = [path_stdmc64]

    for i in range(2, 11):
        print("PointSet: ", i)
        print("steps= 0\nxi=", xi_init)
        path = optimize.robbins_monro(xi_init, functions, params)
        path = optimize.Polyak_Ruppert_averaging(path)
        paths_stdmc64.append(path)
        print()

    ## convergence of the experimental design (Figure 3)

    paths_ = [
        paths_mlmc,
        paths_stdmc1,
        paths_stdmc2,
        paths_stdmc4,
        paths_stdmc8,
        paths_stdmc16,
        paths_stdmc32,
        paths_stdmc64,
    ]
    labels = [
        "MLMC",
        "stdMC(M=1)",
        "stdMC(M=2)",
        "stdMC(M=4)",
        "stdMC(M=8)",
        "stdMC(M=16)",
        "stdMC(M=32)",
        "stdMC(M=64)",
    ]
    inner_samples = [
        2.21,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]
    filename = "convergence_with_path_test"
    visualize_path.convergence_with_path(
        paths_,
        inner_samples,
        labels,
        filename
    )


def expr_pk():

    mlmc_eig.use_reparametrization = True
    mlmc_eig.importance_sampling_method = "Laplace"

    ## Optimization(MLMC)

    print("PK optimization(MLMC)")

    optimize.xi_scale = 24
    max_iters = 10000
    batch_size = 2000
    M0 = 1
    batch_size_eig = 1000000
    M0_eig = 1
    progress_steps_list = [100, 500, 1000, 5000, 10000]
    n_ticks = 24
    params = {"alpha": 0.004, "beta1": 0.9, "beta2": 0.999, "decay_alpha": False}

    condition = optimize.get_condition_num_iters(max_iters)
    restriction = optimize.get_restriction_clip(0, optimize.xi_scale)
    gradient = optimize.get_gradient_randomized_mlmc(
        models.model_pk, mlmc_eig.mlmc_eig_grad, batch_size, M0
    )
    progress = optimize.get_progress_show_xi(
        progress_steps_list, n_ticks, "mlmc_run1"
    )

    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }

    ### first MLMC optimization run (Figure 4 (a))

    print("\nPointSet: ", 1)
    xi_init = np.arange(15) + 1
    print("Begin: ")
    models.show_xi(xi_init, (0, 24), n_ticks, "mlmc_run1-0")
    U_init = mlmc_eig.randomized_mlmc(
        models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init, M0_eig
    ).mean(axis=0)
    print("EIG=", U_init)
    path1_mlmc = optimize.amsgrad(xi_init, functions, params)
    print("End: ")
    U_fin = mlmc_eig.randomized_mlmc(
        models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, path1_mlmc[-1], M0_eig
    ).mean(axis=0)
    print("EIG=", U_fin)
    print()

    #### convergence of variance during optimization (Figure 5)

    num_sample = 100000
    num_level = 10
    mlmc_eig.variance_check_with_path(
        models.model_pk,
        mlmc_eig.mlmc_eig_grad,
        M0,
        num_sample,
        num_level,
        path1_mlmc,
        "var_with_path_pk",
    )

    paths_mlmc = [path1_mlmc]

    ### repeating the same optimization computation 10 times

    functions["progress"] = optimize.progress_pass

    for i in range(2, 11):
        print("PointSet: ", i)
        xi_init = np.arange(15) + 1
        print("Begin: ")
        models.show_xi(xi_init, (0, 24), n_ticks, "mlmc_run" + str(i) + "-0")
        U_init = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init, M0_eig
        ).mean(axis=0)
        print("EIG=", U_init)
        path = optimize.amsgrad(xi_init, functions, params)
        paths_mlmc.append(path)
        print("End: ")
        models.show_xi(
            path[-1], (0, 24), n_ticks, "mlmc_run" + str(i) + "-" + str(max_iters)
        )
        U_fin = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, path[-1], M0
        ).mean(axis=0)
        print("EIG=", U_fin)
        print()

    ## Optimization(stdMC)

    print("PK optimization(stdMC)")

    condition = optimize.get_condition_num_iters(max_iters)
    restriction = optimize.get_restriction_clip(1e-8, optimize.xi_scale)
    gradient = optimize.get_gradient_nested(
        models.model_pk, mlmc_eig.mlmc_eig_grad, batch_size, 1
    )
    progress = optimize.get_progress_show_xi(
        progress_steps_list, n_ticks, "std_run1"
    )

    functions = {
        "condition": condition,
        "restriction": restriction,
        "gradient": gradient,
        "progress": progress,
    }

    ### first standard MC optimization run (Figure 4 (b))

    print("\nPointSet: ", 1)
    xi_init = np.arange(15) + 1
    print("Begin: ")
    models.show_xi(xi_init, (0, 24), n_ticks, "std_run1-0")
    U_init = mlmc_eig.randomized_mlmc(
        models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init, M0_eig
    ).mean(axis=0)
    print("EIG=", U_init)
    path1_std = optimize.amsgrad(xi_init, functions, params)
    print("End: ")
    U_fin = mlmc_eig.randomized_mlmc(
        models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, path1_std[-1], M0_eig
    ).mean(axis=0)
    print("EIG=", U_fin)
    print()

    functions["progress"] = optimize.progress_pass
    paths_std = [path1_std]

    ### repeating the same optimization computation 10 times

    for i in range(2, 11):
        print("PointSet: ", i)
        xi_init = np.arange(15) + 1
        print("Begin: ")
        models.show_xi(xi_init, (0, 24), n_ticks, "std_run" + str(i) + "-0")
        U_init = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, xi_init, M0_eig
        ).mean(axis=0)
        print("EIG=", U_init)
        path = optimize.amsgrad(xi_init, functions, params)
        paths_std.append(path)
        print("End: ")
        models.show_xi(
            path[-1], (0, 24), n_ticks,
            "std_run" + str(i) + "-" + str(max_iters)
        )
        U_fin = mlmc_eig.randomized_mlmc(
            models.model_pk, batch_size_eig, mlmc_eig.mlmc_eig_value, path[-1],  M0
        ).mean(axis=0)
        print("EIG=", U_fin)
        print()

    ## behavior of the expected information gain (Figure 6)

    paths_ = [paths_mlmc, paths_std]
    labels = ["MLMC", "stdMC"]
    M0_eig = 1
    num_step = 500
    num_sample = 1000000
    visualize_eig.eig_mean_with_error(
        paths_, labels, num_step, None, "eig_mean_with_error", M0_eig, num_sample
    )
