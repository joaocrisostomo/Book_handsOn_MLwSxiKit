"""A simple demonstration of how to solve a symbolic regression problem
using efficient implementation of GSGP algorithm. This example served as
a basis for the paper.
"""


def main():
    import os
    import datetime

    import torch

    from gpol.algorithms.genetic_algorithm import GSGP
    from gpol.problems.inductive_programming import SMLGS
    from gpol.operators.initializers import rhh, prm_grow
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_efficient_gs_xo, prm_efficient_gs_mtn

    from gpol.utils.utils import rmse, train_test_split
    from gpol.utils.inductive_programming import function_map
    from gpol.utils.datasets import load_boston

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to create an instance of SML problem for GSGP
    # Defines the processing device and the random state 's seed
    device, seed = "cpu", 0
    # Characterizes functions' set
    f_set = [function_map["add"], function_map["sub"], function_map["mul"], function_map["div"]]
    # Characterizes constants' set
    start_c, end_c, size_c = -2.0, 2.0, 10.0
    c_set = torch.arange(start=start_c, end=end_c, step=abs(end_c-start_c)/size_c, device=device)
    # Defines data's usage properties
    p_test, batch_size = 0.3, 50
    # Loads the data and puts on the processing device
    X, y = load_boston(X_y=True)
    X = X.to(device)
    y = y.to(device)
    # Creates the solve space
    sspace = {"n_dims": X.shape[1], "function_set": f_set, "p_constants": 0.1, "constant_set": c_set,
              "max_init_depth": 5}
    # Partitions the data
    train_indices, test_indices = train_test_split(X=X, y=y, p_test=p_test, shuffle=True, indices_only=True, seed=seed)
    # Creates problem instance for running Geometric Semantic Operators without trees' construction
    pi = SMLGS(sspace=sspace, ffunction=rmse, X=X, y=y, train_indices=train_indices, test_indices=test_indices,
               batch_size=batch_size, min_=True)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    pop_size, n_iter, = 100, 30
    # Defines algorithm's parameters
    single_init = prm_grow(sspace)  # creates single trees' initializer
    to, by = 5.0, 0.25  # defines mutation's steps
    p_m, ms = 0.3, torch.arange(by, to + by, by, device=device)  # GSM's probability and mutation's step
    gsgp_pars = {"initializer": rhh, "selector": prm_tournament(pressure=0.1),
                 "mutator": prm_efficient_gs_mtn(X, single_init, ms), "crossover": prm_efficient_gs_xo(X, single_init),
                 "p_m": p_m, "p_c": 1.0 - p_m, "pop_size": pop_size, "elitism": True, "reproduction": False}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Setup the logging properties.
    # Creates the logger
    experiment_label = "SML-IPOP"  # SML approached from the perspective of Inductive Programming
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Defines a connection string to store random trees
    path_rts = os.path.join(path, "reconstruct", "rts")
    if not os.path.exists(path_rts):
        os.makedirs(path_rts)
    # Defines a connection string to store the initial population
    path_init_pop = os.path.join(path, "reconstruct", "init_pop")
    if not os.path.exists(path_init_pop):
        os.makedirs(path_init_pop)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Executes GSG
    # Creates an instance of GSGP solve algorithm
    isa = GSGP(pi=pi, seed=seed, device=device, path_init_pop=path_init_pop, path_rts=path_rts, **gsgp_pars)
    # Solves the PI
    isa.solve(n_iter=n_iter, tol=0.5, n_iter_tol=5, test_elite=True, verbose=2)
    print("Best training fitness: {:.3f}".format(isa.best_sol.fit))
    print("Best test fitness: {:.3f}".format(isa.best_sol.test_fit))
    print("Best solution's depth:", isa.best_sol.depth)
    isa.write_history(os.path.join(path, "reconstruct", experiment_label + "_seed_" + str(seed) + "_history.csv"))


if __name__ == "__main__":
    main()
