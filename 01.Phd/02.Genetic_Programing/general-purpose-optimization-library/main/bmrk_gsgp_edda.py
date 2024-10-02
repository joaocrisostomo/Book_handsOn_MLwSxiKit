""" Examples
An exhaustive benchmark of the implemented algorithms on few sample
problems. The procedure produces summary tables with aggregated results.
"""


def main():
    import os
    import time
    import logging
    import datetime

    import pandas as pd

    import torch

    from gpol.algorithms.genetic_algorithm import GSGP
    from gpol.problems.inductive_programming import SMLGS
    from gpol.operators.initializers import prm_full, prm_edda
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_efficient_gs_xo, prm_efficient_gs_mtn
    from gpol.utils.inductive_programming import function_map
    from gpol.utils.utils import rmse, train_test_split
    from gpol.utils.datasets import load_diabetes, load_boston

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > >´Setup the resources and the parameters
    # Defines the computational resources
    device, n_runs, pop_size, n_iter, p_test = 'cpu', 2, 50, 10, 0.3
    # Characterizes data manipulation's parameters
    batch_size, shuffle, n_jobs = 100, True, 1
    # Defines the problems
    problems = {"Boston": load_boston, "Diabetes": load_diabetes}
    # Defines the function set
    fset = [function_map["add"], function_map["sub"], function_map["mul"], function_map["div"], function_map["log"],
            function_map["sin"], function_map["cos"], function_map["mean"], function_map["max"], function_map["min"]]
    # Defines the constant set
    start_c, end_c, size_c = -2.0, 2.0, 10.0
    cset = torch.arange(start=start_c, end=end_c, step=abs(end_c - start_c) / size_c, device=device)
    # Creates the solve space
    sspace = {"function_set": fset, "p_constants": 0.1, "constant_set": cset, "max_init_depth": 5, "n_batches": 1}
    # Defines single trees' initializer
    single_initializer = prm_full(sspace)
    # Defines geometric mutation's parameters
    to, by = 5.0, 0.25
    p_m, ms, pressure, = 0.3, torch.arange(by, to + by, by, device=device), 0.1
    gsgp_pars = {"selector": prm_tournament(pressure=0.05), "p_m": p_m, "p_c": 1.0 - p_m, "pop_size": pop_size,
                 "elitism": True, "reproduction": False}
    # Defines EDDA parameters
    edda_pars = {"deme_size": 50, "maturation": 5, "p_gp": 0.5, "p_features": 0.5, "p_samples": 0.5, "p_functions": 0.5,
                 "replacement": True, "bnd_pressure": (0.05, 0.2), "ms": ms, "batch_size": None, "shuffle": True,
                 "ffunction": rmse, "min_": True, "verbose": 2, "log": 2, "n_jobs": 8}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > >´Setup logging properties.
    # Creates the logger
    experiment_label = "SMLGS-IPOP-EDDA"  # SML approached from the perspective of Inductive Programming with EDDA
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=os.path.join(path, "log" + ".txt"), level=logging.DEBUG, format='%(name)s,%(message)s')
    # Defines a connection string to store random trees
    path_rts = os.path.join(path, "reconstruct", "rts")
    if not os.path.exists(path_rts):
        os.makedirs(path_rts)
    # Defines a connection string to store the initial population
    path_init_pop = os.path.join(path, "reconstruct", "init_pop")
    if not os.path.exists(path_init_pop):
        os.makedirs(path_init_pop)
    # Creates a containers for the summary statistics
    summary_dict = {"OP": [], "D": [], "ISA": [], "Run": [], "Time": [], "TestFit": [], "Fit": [],
                    "MeanFit": [], "STDFit": []}
    # Defines aggregations' dict
    agg_dict = {"Time": ['mean', 'std'], "Fit": ['mean', 'std'], "TestFit": ['mean', 'std'],
                "MeanFit": ['mean', 'std'], "STDFit": ['mean', 'std']}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Runs the benchmark
    for prob_name, loader in problems.items():
        # Prints problem's name
        print("|||>>>>>>>>>||| PROBLEM: {} |||<<<<<<<<<|||".format(prob_name))
        # Loads the data and puts on the processing device
        X, y = loader(X_y=True)
        X = X.to(device)
        y = y.to(device)
        # Completes the solve space with the number of features
        sspace["n_dims"] = X.shape[1]
        # Executes the algorithms
        for seed in range(n_runs):
            start_time = time.time()
            # Partitions the data
            train_indices, test_indices = train_test_split(X=X, y=y, p_test=p_test, shuffle=True, indices_only=True,
                                                           seed=seed+5)

            # Creates problem's instance for running Geometric Semantic Operators without trees' construction
            pi_gsgp = SMLGS(sspace=sspace, ffunction=rmse, X=X, y=y, train_indices=train_indices,
                            test_indices=test_indices, batch_size=batch_size, min_=True)

            # Adds initializer to gsgp_pars
            gsgp_pars["initializer"] = prm_edda(**edda_pars, X=X, y=y, train_indices=train_indices,
                                                test_indices=test_indices)

            # Adds mutator to gsgp_pars
            gsgp_pars["mutator"] = prm_efficient_gs_mtn(X, single_initializer, ms)
            # Adds crossover to gsgp_pars
            gsgp_pars["crossover"] = prm_efficient_gs_xo(X, single_initializer)
            # Creates an instance of a solve algorithm
            isa = GSGP(pi=pi_gsgp, seed=seed, device=device, **gsgp_pars)
            # Solves the PI
            isa.solve(n_iter=n_iter, test_elite=True, verbose=2, log=2)
            print("Best training fitness: {:.3f}".format(isa.best_sol.fit))
            print("Best test fitness: {:.3f}".format(isa.best_sol.test_fit))
            print("Best solution's depth:", isa.best_sol.depth)
            isa.write_history(os.path.join(path, "reconstruct", experiment_label + "_seed_" + str(seed) + "_history.csv"))
            # Writes summary statistics
            summary_dict["OP"].append(prob_name)
            summary_dict["D"].append(sspace["n_dims"])
            summary_dict["ISA"].append("GSGP")
            summary_dict["Run"].append(seed)
            summary_dict["Time"].append(time.time() - start_time)
            summary_dict["TestFit"].append(isa.best_sol.test_fit.item() if hasattr(isa.best_sol, 'test_fit') else -1.0)
            summary_dict["Fit"].append(isa.best_sol.fit.item())
            summary_dict["MeanFit"].append(isa.pop.fit.mean().item() if hasattr(isa, 'pop') else -1.0)
            summary_dict["STDFit"].append(isa.pop.fit.std().item() if hasattr(isa, 'pop') else -1.0)

    # Creates and writes the summary statistics table
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.to_csv(os.path.join(path, "stats.txt"), index=False)

    # Creates a grouped (by run and problem) summary statistics
    summary_df.groupby(["ISA", "OP"]).agg(agg_dict).to_csv(os.path.join(path, "stats_grouped.txt"), index=True)


if __name__ == "__main__":
    main()
