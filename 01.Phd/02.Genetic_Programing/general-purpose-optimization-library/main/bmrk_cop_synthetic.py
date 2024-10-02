"""Examples.
An exhaustive benchmark of the implemented algorithms on few problems
from the field of continuous optimization. The procedure produces
summary tables with aggregated results.
"""


def main():
    import os
    import time
    import logging
    import datetime

    import torch
    import pandas as pd

    from gpol.problems.continuous import Box
    from gpol.problems.utils import sphere_function, rastrigin_function, ackley_function, rosenbrock_function

    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing, SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.algorithms.differential_evolution import DifferentialEvolution
    from gpol.algorithms.swarm_intelligence import SPSO, APSO
    from gpol.operators.initializers import rnd_muniform, rnd_vuniform
    from gpol.operators.selectors import prm_tournament, prm_dernd_selection
    from gpol.operators.variators import prm_iball_mtn, geometric_xo, prm_pso, \
        de_binomial_xo, de_exponential_xo, de_target_to_best, de_best

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines Optimization Problems' parameters
    # Defines the processing device
    device = "cpu"
    # Defines the number of dimensions for the optimization problems (OPs)
    n_dims = [2, 30]
    # Defines the set of synthetic OPs
    problems = {"Rosenbrock": rosenbrock_function,
                "Rastrigin": rastrigin_function,
                "Ackley": ackley_function,
                "Sphere": sphere_function}
    # Defines the OPs' constraints
    constraints = {"Rosenbrock": torch.tensor([-2.048, 2.048], device=device),
                   "Rastrigin": torch.tensor([-5.12, 5.12], device=device),
                   "Ackley": torch.tensor([-32.768, 32.786], device=device),
                   "Sphere": torch.tensor([-5.12, 5.12], device=device)}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    n_runs, pop_size, n_iter = 30, 50, 100
    # Defines RS's parameters
    rs_pars = {"initializer": rnd_vuniform}
    # Defines HC's parameters
    p_m_ball, radius = 0.5, 0.3
    mutator = prm_iball_mtn(0.5, radius)
    hc_pars = {"initializer": rnd_vuniform, "nh_function": mutator, "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    sa_pars = {"initializer": rnd_vuniform, "nh_function": mutator, "control": control, "update_rate": update_rate,
               "nh_size": pop_size}
    # Defines GA's parameters
    p_m = 0.3  # mutation's probability
    ga_pars = {"initializer": rnd_muniform, "selector": prm_tournament(pressure=0.08), "mutator": mutator, "p_m": p_m,
               "crossover": geometric_xo, "p_c": 1.0-p_m, "pop_size": pop_size, "elitism": True, "reproduction": False}
    # Defines SPSO's parameters
    social_f, cognitive_f, w_max, w_min, v_clamp = 2.0, 2.0, 0.9, 0.4, 3.0
    # Defines modified SPSO's parameters
    pso_update_rule = prm_pso(c1=social_f, c2=cognitive_f, w_max=w_max, w_min=w_min)
    pso_pars = {"initializer": rnd_muniform, "mutator": pso_update_rule, "v_clamp": v_clamp, "pop_size": pop_size}
    # Defines DE/best/1/exp
    n_parents, m_weights, c_rate = 2, torch.tensor([0.7, 0.7], device=device), 0.5
    de_best_1_exp = {"initializer": rnd_muniform, "selector": prm_dernd_selection(n_sols=n_parents),
                     "mutator": de_target_to_best, "crossover": de_exponential_xo, "m_weights": m_weights,
                     "c_rate": c_rate, "pop_size": pop_size}
    # Defines DE/best/2/bin
    n_parents, m_weights = 4, torch.tensor([0.7, 0.7], device=device)
    de_best_2_bin = {"initializer": rnd_muniform, "selector": prm_dernd_selection(n_sols=n_parents), "mutator": de_best,
                     "crossover": de_binomial_xo, "m_weights": m_weights, "c_rate": c_rate, "pop_size": pop_size}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Setup the logging properties
    # Creates the logger
    experiment_label = "SCOP"  # Syntetic Continuous Optimization Problems
    time_id = str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().hour) + "_" + \
              str(datetime.datetime.now().minute) + "_" + str(datetime.datetime.now().second)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logFiles", experiment_label + "_" + time_id)
    # Creates directory if does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=os.path.join(path, "log" + ".txt"), level=logging.DEBUG, format='%(name)s,%(message)s')
    # Creates containers for the summary statistics
    fit, fit_avg, fit_std, runs, algorithms, problems_, n_dims_, timing = [], [], [], [], [], [], [], []
    # Defines aggregations' dict
    agg_dict = {"Fit": ['mean', 'std'], "Time": ['mean', 'std'], "MeanFit": ['mean', 'std'], "STDFit": ['mean', 'std']}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Runs the benchmark
    for problem, p_constraints in zip(problems.values(), constraints.values()):
        for d in n_dims:
            # Creates a problem's instance (PI)
            pi = Box(sspace={"n_dims": d, "constraints": p_constraints}, ffunction=problem, min_=True, bound=True)
            pi.__name__ = problem.__name__
            for seed in range(n_runs):
                print("-" * 75)
                print("--- PROBLEM {} \t DIMENSION {} \t RUN {}".format(problem.__name__, d, seed))
                print("-" * 75)

                # Crates and stores algorithms' instances in a dictionary
                isas = {"RS": RandomSearch(pi=pi, seed=seed, device=device, **rs_pars),
                        "HC": HillClimbing(pi=pi, seed=seed, device=device, **hc_pars),
                        "SA": SimulatedAnnealing(pi=pi, seed=seed, device=device, **sa_pars),
                        "GA": GeneticAlgorithm(pi=pi, seed=seed, device=device, **ga_pars),
                        "DE/best/1/exp": DifferentialEvolution(pi=pi, seed=seed, device=device, **de_best_1_exp),
                        "DE/best/2/bin": DifferentialEvolution(pi=pi, seed=seed, device=device, **de_best_2_bin),
                        "S-PSO": SPSO(pi=pi, seed=seed, device=device, **pso_pars),
                        "A-PSO": APSO(pi=pi, seed=seed, device=device, **pso_pars)}

                # Loops over the algorithms' instances
                for isa_name, isa in isas.items():
                    print("|||>>>>>>>>>||| {} |||<<<<<<<<<|||".format(isa_name))
                    start_time = time.time()

                    # Updates algorithms' name for the log file
                    isa.__name__ = isa_name

                    # Updates mutation' parameters for a larger amount of dimensions
                    mutator = prm_iball_mtn(0.5 if d == 2 else 0.2, radius)
                    if isa_name in ["HC", "SA", "GA"]:
                        isa.mutator = mutator

                    # Computes computational effort for the Random Search
                    n_iter_ = n_iter * pop_size if isa_name == "RS" else n_iter

                    # Solves the PI
                    isa.solve(n_iter=n_iter_, verbose=1 if isa_name == "RS" else 2, log=1 if isa_name == "RS" else 2)
                    print("Best solution: \t training fitness = {:.3f}".format(isa.best_sol.fit, isa.best_sol.repr_))
                    fit.append(isa.best_sol.fit.item())
                    fit_avg.append(isa.pop.fit.mean().item() if hasattr(isa, 'pop') else -1.0)
                    fit_std.append(isa.pop.fit.std().item() if hasattr(isa, 'pop') else -1.0)
                    runs.append(seed)
                    algorithms.append(isa_name)
                    problems_.append(problem.__name__)
                    n_dims_.append(d)
                    timing.append(time.time() - start_time)

    # Creates summary statistics table
    summary_df = pd.DataFrame.from_dict({"Run": runs, "ISA": algorithms, "OP": problems_, "D": n_dims_, "Time": timing,
                                         "Fit": fit, "MeanFit": fit_avg, "STDFit": fit_std})
    summary_df.to_csv(os.path.join(path, "stats.txt"), index=False)
    # Grouped (by run) summary statistics
    summary_df.groupby(["ISA", "OP", "D"]).agg(agg_dict).to_csv(os.path.join(path, "stats_grouped.txt"), index=True)


if __name__ == "__main__":
    main()
