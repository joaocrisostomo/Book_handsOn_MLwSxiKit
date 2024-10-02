"""A simple demonstration of how to solve a Travelling Salesman Problem
(TSP) using GPOL's algorithms that are appropriate for this kind of
optimization problem. This example served as a basis for the paper.
"""


def main():
    import torch

    from gpol.problems.tsp import TSP
    from gpol.utils.utils import travel_distance
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing
    from gpol.algorithms.local_search import SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.operators.initializers import rnd_vshuffle, rnd_mshuffle
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import partially_mapped_xo, prm_iswap_mnt

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to create an instance of TSP COP
    # Defines the processing device and the random state 's seed
    device, seed = 'cpu', 0
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
    # Creates an instance of optimization problem (PI)
    pi = TSP(sspace=sspace, ffunction=travel_distance, min_=True)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    pop_size, n_iter = 100, 30
    # Defines RS's parameters
    pars = {RandomSearch: {"initializer": rnd_vshuffle}}
    # Defines HC's parameters
    p_m = 0.2  # probability of swapping a given index in the chromosome
    pars[HillClimbing] = {"initializer": rnd_vshuffle, "nh_function": prm_iswap_mnt(p_m), "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    pars[SimulatedAnnealing] = {"initializer": rnd_vshuffle, "nh_function": prm_iswap_mnt(p_m), "nh_size": pop_size,
                                "control": control, "update_rate": update_rate}
    # Defines GA's parameters
    p_m_ga = 0.3  # GA's probability of mutation
    pars[GeneticAlgorithm] = {"pop_size": pop_size, "initializer": rnd_mshuffle,
                              "selector": prm_tournament(pressure=0.1), "mutator": prm_iswap_mnt(p_m),
                              "crossover": partially_mapped_xo, "p_m": p_m_ga, "p_c": 1.0 - p_m_ga, "elitism": True,
                              "reproduction": False}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Iterates algorithms and their respective parameters
    for isa_type, isa_pars in pars.items():
        # Creates an instance of a solve algorithm
        isa = isa_type(pi=pi, **isa_pars, seed=seed, device=device)
        # n_iter*pop_size if isinstance(isa, RandomSearch) else n_iter
        # Solves the PI
        isa.solve(n_iter=n_iter, tol=20, n_iter_tol=5, verbose=2)
        print("Algorithm: {}".format(isa_type.__name__))
        print("Best solution's fitness: {:.3f}".format(isa.best_sol.fit))
        print("Best solution:", isa.best_sol.repr_)


if __name__ == "__main__":
    main()
