""" Examples
An exhaustive benchmark of the implemented algorithms on the Travelling
Salesman combinatorial optimization problem. The procedure produces
summary tables with aggregated results.
"""


def main():
    import os
    import time
    import logging
    import datetime

    import pandas as pd
    import torch

    from gpol.problems.tsp import TSP
    from gpol.utils.utils import travel_distance
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing
    from gpol.algorithms.local_search import SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.operators.initializers import rnd_vshuffle, rnd_mshuffle
    from gpol.operators.selectors import rank_selection
    from gpol.operators.variators import partially_mapped_xo, prm_iswap_mnt

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines Optimization Problems' parameters
    # Defines the processing device
    device = "cpu"
    # Characterizes the problem: distance matrix (from https://developers.google.com/optimization/routing/tsp)
    dist_mtrx = torch.tensor([[0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
                [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
                [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
                [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
                [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
                [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
                [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
                [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
                [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
                [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
                [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
                [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
                [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]], dtype=torch.float, device=device)
    # Creates the solve space
    sspace = {"distances": dist_mtrx, "origin": 3}
    # Creates the problem's instance (PI)
    pi = TSP(sspace=sspace, ffunction=travel_distance, min_=True)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    n_runs, pop_size, n_iter = 30, 500, 100
    # Defines RS's parameters
    rs_pars = {"initializer": rnd_vshuffle}
    # Defines HC's parameters
    p_m_swap = 0.2  # probability of swapping at a given index in the chromosome
    mutator = prm_iswap_mnt(prob=p_m_swap)
    hc_pars = {"initializer": rnd_vshuffle, "nh_function": mutator, "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    sa_pars = {"initializer": rnd_vshuffle, "nh_function": mutator, "control": control, "update_rate": update_rate,
               "nh_size": pop_size}
    # Defines GA's parameters
    p_m = 0.3
    ga_pars = {"pop_size": pop_size, "initializer": rnd_mshuffle, "selector": rank_selection, "mutator": mutator,
               "crossover": partially_mapped_xo, "p_m": p_m, "p_c": 1-p_m, "pop_size": pop_size, "elitism": True,
               "reproduction": False}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > >´Setup the logging properties
    # Creates the logger
    experiment_label = "TSP"
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=os.path.join(path, "log" + ".txt"), level=logging.DEBUG, format='%(name)s,%(message)s')
    # Creates a containers for the summary statistics
    summary_dict = {"OP": [], "D": [], "ISA": [], "Run": [], "Time": [], "Fit": [], "MeanFit": [], "STDFit": []}
    # Defines aggregations' dict
    agg_dict = {"Fit": ['mean', 'std'], "Time": ['mean', 'std'], "MeanFit": ['mean', 'std'], "STDFit": ['mean', 'std']}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Runs the benchmark
    for seed in range(n_runs):
        print("-" * 75)
        print("--- PROBLEM: {} \t RUN: {}".format(experiment_label, int(seed)))
        print("-" * 75)

        # Crates and stores algorithms' instances in a dictionary
        isas = {"RS": RandomSearch(pi=pi, seed=seed, device=device, **rs_pars),
                "HC": HillClimbing(pi=pi, seed=seed, device=device, **hc_pars),
                "SA": SimulatedAnnealing(pi=pi, seed=seed, device=device, **sa_pars),
                "GA": GeneticAlgorithm(pi=pi, seed=seed, device=device, **ga_pars)}

        # Loops over the algorithms' instances
        for isa_name, isa in isas.items():
            print("|||>>>>>>>>>||| {} |||<<<<<<<<<|||".format(isa_name))
            start_time = time.time()
            # RS's equivalency: n_iter * pop_size
            n_iter_ = n_iter * pop_size if isa_name == "RS" else n_iter
            # Solves the PI
            isa.solve(n_iter=n_iter_, verbose=2, log=2)
            print("<><><> Best solution's path ", isa.best_sol.repr_)
            print("Best solution: \t fitness = {:.3f}".format(isa.best_sol.fit.item()))
            summary_dict["OP"].append(experiment_label)
            summary_dict["D"].append(len(dist_mtrx))
            summary_dict["ISA"].append(isa_name)
            summary_dict["Run"].append(seed)
            summary_dict["Time"].append(time.time() - start_time)
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
