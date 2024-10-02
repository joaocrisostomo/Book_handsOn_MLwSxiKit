""" Examples
An exhaustive benchmark of the implemented algorithms on few symbolic
regression problems. The procedure produces summary tables with
aggregated results.
"""


def main():
    import os
    import time
    import logging
    import datetime

    import pandas as pd

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    from gpol.problems.inductive_programming import SML
    from gpol.operators.initializers import rhh, prm_full, grow
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_gs_xo, prm_gs_mtn, swap_xo, prm_subtree_mtn
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing, SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm

    from gpol.utils.utils import rmse, train_test_split
    from gpol.utils.datasets import load_diabetes, load_boston
    from gpol.utils.inductive_programming import function_map

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines Optimization Problems' parameters
    # Defines the processing device
    device = "cpu"
    # Defines the function set
    fset = [function_map["add"], function_map["sub"], function_map["mul"], function_map["div"]]
    # Defines the constant set
    start_c, end_c, size_c = -2.0, 2.0, 10.0
    cset = torch.arange(start=start_c, end=end_c, step=abs(end_c - start_c) / size_c, device=device)
    # Creates the solve space
    sspace = {"function_set": fset, "constant_set": cset, "p_constants": 0.1, "max_init_depth": 5, "max_depth": -1,
              "n_batches": 1}
    # Defines the problems
    problems = {"Boston": load_boston, "Diabetes": load_diabetes}
    # Characterizes data manipulation parameters
    p_test, batch_size, shuffle, n_jobs = 0.3, 50, True, 1

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    n_runs, pop_size, n_iter = 30, 500, 100
    # Defines single trees' initializer
    single_initializer = prm_full(sspace)
    # Defines RS's parameters
    rs_pars = {"initializer": grow}
    # Defines HC-GP's parameters
    mutator_gp = prm_subtree_mtn(single_initializer)
    hc_gp_pars = {"initializer": grow, "nh_function": mutator_gp, "nh_size": pop_size}
    # Defines HC-GSGP's parameters
    to, by = 5.0, 0.25  # GSM's range of mutation steps
    ms = torch.arange(by, to + by, by, device=device)  # GSM's mutation steps
    mutator_gsgp = prm_gs_mtn(single_initializer, ms)
    hc_gsgp_pars = {"initializer": grow, "nh_function": mutator_gsgp, "nh_size": pop_size}
    # Defines SA-GP's parameters
    control, update_rate = 1.0, 0.9
    sa_gp_pars = {"initializer": grow, "nh_function": mutator_gp, "nh_size": pop_size, "control": control,
                  "update_rate": update_rate}
    # Defines SA-GSGP's parameters
    sa_gsgp_pars = {"initializer": grow, "nh_function": mutator_gsgp, "nh_size": pop_size, "control": control,
                    "update_rate": update_rate}
    # Defines GP's parameters
    selector = prm_tournament(pressure=0.08)
    p_m = 0.3
    ga_gp_pars = {"initializer": rhh, "selector": selector, "mutator": mutator_gp, "crossover": swap_xo, "p_m": p_m,
                  "p_c": 1.0-p_m, "pop_size": pop_size, "elitism": True, "reproduction": False}
    # Defines GSGP's parameters
    p_m_gs = 1.0
    ga_gsgp_pars = {"initializer": rhh, "selector": selector,  "mutator": mutator_gsgp,
                    "crossover": prm_gs_xo(single_initializer, device), "p_m": p_m_gs, "p_c": 1.0-p_m_gs,
                    "pop_size": pop_size, "elitism": True, "reproduction": False}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Setup logging properties.
    # Creates the logger
    experiment_label = "SML-IPOP"  # SML approached from the perspective of Inductive Programming
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=os.path.join(path, "log" + ".txt"), level=logging.DEBUG, format='%(name)s,%(message)s')
    # Creates a containers for the summary statistics
    summary_dict = {"OP": [], "D": [], "ISA": [], "Run": [], "Time": [], "TestFit": [], "Fit": [],
                    "MeanFit": [], "STDFit": []}
    # Defines aggregations' dict
    agg_dict = {"Time": ['mean', 'std'], "Fit": ['mean', 'std'], "TestFit": ['mean', 'std'],
                "MeanFit": ['mean', 'std'], "STDFit": ['mean', 'std']}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Runs the benchmark
    for prob_name, loader in problems.items():
        print("|||>>>>>>>>>||| PROBLEM: {} |||<<<<<<<<<|||".format(prob_name))
        # Loads the data
        X, y = loader(X_y=True)
        # Completes the solve space with the number of features (a.k.a. dimensions)
        sspace["n_dims"] = X.shape[1]
        for seed in range(n_runs):
            # Splits the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.3, seed=seed)
            # Creates two objects of type TensorDataset: one for training, another for test data
            ds_train = TensorDataset(X_train, y_train)
            ds_test = TensorDataset(X_test, y_test)
            # Creates training and test DataLoader
            batch_size = X_train.shape[0]  # NOTE: comment this line to run it in batch mode
            dl_train = DataLoader(ds_train, batch_size, shuffle)
            dl_test = DataLoader(ds_test, batch_size, shuffle)
            # Creates a problem's instance (PI)
            pi = SML(sspace, rmse, dl_train, dl_test, n_jobs=n_jobs, min_=True)

            # Crates and stores algorithms' instances in a dictionary
            isas = {"RS": RandomSearch(pi=pi, seed=seed, device=device, **rs_pars),
                    "HC-GP": HillClimbing(pi=pi, seed=seed, device=device, **hc_gp_pars),
                    "HC-GSGP": HillClimbing(pi=pi, seed=seed, device=device, **hc_gsgp_pars),
                    "SA-GP": SimulatedAnnealing(pi=pi, seed=seed, device=device, **sa_gp_pars),
                    "SA-GSGP": SimulatedAnnealing(pi=pi, seed=seed, device=device, **sa_gsgp_pars),
                    "GP": GeneticAlgorithm(pi=pi, seed=seed, device=device, **ga_gp_pars),
                    "GSGP": GeneticAlgorithm(pi=pi, seed=seed, device=device, **ga_gsgp_pars)}

            # Loops over the algorithms' instances
            for isa_name, isa in isas.items():
                print("|||>>>>>>>>>||| ALGORITHM: {} |||<<<<<<<<<|||".format(isa_name))
                isa.__name__ = isa_name
                start_time = time.time()
                # Readjust the solve-space for GSGP
                sspace["max_depth"] = -1 if "GSGP" in isa_name else 15
                # RS's equivalency: n_iter * pop_size
                n_iter_ = n_iter * pop_size if isa_name == "RS" else n_iter
                # Solves the PI
                isa.solve(n_iter=n_iter_, start_at=None, test_elite=True, verbose=2, log=1 if isa_name == "RS" else 2)
                print("Best solution: \t training fit={0:.3f} \t test fit={1:.3f}".format(isa.best_sol.fit,
                                                                                          isa.best_sol.test_fit,
                                                                                          isa.best_sol.repr_))
                summary_dict["OP"].append(prob_name)
                summary_dict["D"].append(sspace["n_dims"])
                summary_dict["ISA"].append(isa_name)
                summary_dict["Run"].append(seed)
                summary_dict["Time"].append(time.time() - start_time)
                summary_dict["TestFit"].append(isa.best_sol.test_fit.item() if hasattr(isa.best_sol, 'test_fit') else -1)
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
