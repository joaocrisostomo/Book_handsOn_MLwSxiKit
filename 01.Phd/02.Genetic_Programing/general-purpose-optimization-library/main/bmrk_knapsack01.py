""" Examples
An exhaustive benchmark of the implemented algorithms on the so-called
binary knapsack combinatorial optimization problem. The procedure
produces summary tables with aggregated results.
"""


def main():
    import os
    import time
    import logging
    import datetime

    import torch
    import pandas as pd

    from gpol.problems.knapsack import Knapsack01

    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing
    from gpol.algorithms.local_search import SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.operators.initializers import prm_rnd_mint, prm_rnd_vint
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_ibinary_flip, one_point_xo

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines Optimization Problems' parameters
    # Defines the processing device
    device = "cpu"
    # Characterizes the 0-1 Knapsack problem
    capacity, n_items, min_ = 1000, 100, False
    # Defines the lower and the upper bounds for weights' and values' generation
    lb, ub = 0, 10

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    n_runs, pop_size, n_iter = 30, 500, 100
    # Defines RS's parameters
    rs_pars = {"initializer": prm_rnd_vint()}
    # Defines HC's parameters
    p_m_flip = 0.2  # probability of flipping at a given index in the chromosome
    mutator = prm_ibinary_flip(prob=p_m_flip)
    hc_pars = {"initializer": prm_rnd_vint(), "nh_function": mutator, "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    sa_pars = {"initializer": prm_rnd_vint(), "nh_function": mutator, "control": control, "update_rate": update_rate,
               "nh_size": pop_size}
    # Defines GA's parameters
    p_m = 0.3
    ga_pars = {"initializer": prm_rnd_mint(), "selector": prm_tournament(pressure=0.08), "mutator": mutator,
               "crossover": one_point_xo, "p_m": p_m, "p_c": 1.0-p_m, "pop_size": pop_size, "elitism": True,
               "reproduction": False}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Setup the logging properties
    # Creates the logger
    experiment_label = "Knapsack01"
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=os.path.join(path, "log" + ".txt"), level=logging.DEBUG, format='%(name)s,%(message)s')
    # Creates a containers for the summary statistics
    summary_dict = {"OP": [], "D": [], "ISA": [], "Run": [], "Time": [], "Fit": [], "Titems": [], "Tweight": [],
                    "MeanFit": [], "STDFit": []}
    # Defines aggregations' dict
    agg_dict = {"Fit": ['mean', 'std'], "Titems": ['mean', 'std'], "Tweight": ['mean', 'std'], "Time": ['mean', 'std'],
                "MeanFit": ['mean', 'std'], "STDFit": ['mean', 'std']}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Runs the benchmark
    for seed in range(n_runs):
        print("-" * 75)
        print("--- PROBLEM: {} \t RUN: {}".format(experiment_label, int(seed)))
        print("-" * 75)
        # Sets the random seed (necessary only for this specific experiment because items are generated randomly)
        torch.manual_seed(seed)

        # Creates the solve-space
        sspace = {"n_dims": n_items, "capacity": capacity,
                  "weights": torch.FloatTensor(n_items).uniform_(lb, ub).to(device),
                  "values": torch.FloatTensor(n_items).uniform_(lb, ub).to(device)}
        # Creates the problem's instance (PI)
        pi = Knapsack01(sspace, torch.matmul, min_)

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
            # Auxiliary
            titems, tweight = isa.best_sol.repr_.sum().item(), torch.dot(isa.best_sol.repr_, sspace["weights"]).item()
            print("Best solution: \t fitness = {:.3f} \t weight {:.3f} \t items {:.3f}".format(isa.best_sol.fit.item(),
                                                                                               tweight, titems))
            summary_dict["OP"].append(experiment_label)
            summary_dict["D"].append(sspace["n_dims"])
            summary_dict["ISA"].append(isa_name)
            summary_dict["Run"].append(seed)
            summary_dict["Time"].append(time.time() - start_time)
            summary_dict["Fit"].append(isa.best_sol.fit.item())
            summary_dict["Titems"].append(titems)
            summary_dict["Tweight"].append(tweight)
            summary_dict["MeanFit"].append(isa.pop.fit.mean().item() if hasattr(isa, 'pop') else -1.0)
            summary_dict["STDFit"].append(isa.pop.fit.std().item() if hasattr(isa, 'pop') else -1.0)

    # Creates and writes the summary statistics table
    summary_df = pd.DataFrame.from_dict(summary_dict)
    summary_df.to_csv(os.path.join(path, "stats.txt"), index=False)

    # Creates a grouped (by run and problem) summary statistics
    summary_df.groupby(["ISA", "OP"]).agg(agg_dict).to_csv(os.path.join(path, "stats_grouped.txt"), index=True)


if __name__ == "__main__":
    main()
