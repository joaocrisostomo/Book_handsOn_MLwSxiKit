"""A simple demonstration of how to solve a Bounded Knapsack problem
using GPOL's algorithms that are appropriate for this kind of
optimization problem. This example served as a basis for the paper.
"""


def main():
    import torch
    from gpol.problems.knapsack import KnapsackBounded
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing
    from gpol.algorithms.local_search import SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.operators.initializers import prm_rnd_mint, prm_rnd_vint
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_rnd_int_ibound, one_point_xo

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to create an instance of TSP COP
    # Defines the processing device and the random state 's seed
    device, seed = 'cuda' if torch.cuda.is_available() else 'cpu', 0
    # Characterizes the problem: defines the number of items and their maximum allowed repetitions
    n_items, max_rep = 17, 4
    # Creates the search space
    torch.manual_seed(seed)  # set seed for the random values generation
    sspace = {"capacity": 160, "n_dims": n_items, "weights": torch.FloatTensor(n_items).uniform_(1, 9).to(device),
              "values": torch.FloatTensor(n_items).uniform_(0.5, 20).to(device),
              "bounds": torch.stack((torch.zeros(n_items), max_rep * torch.ones(n_items))).to(device)}
    # Creates an instance of optimization problem (PI)
    pi = KnapsackBounded(sspace=sspace, ffunction=torch.matmul, min_=False)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    pop_size, n_iter = 100, 30
    # Defines RS's parameters
    pars = {RandomSearch: {"initializer": prm_rnd_vint(0, max_rep+1)}}
    # Defines HC's parameters
    mutator = prm_rnd_int_ibound(prob=0.5, lb=0, ub=max_rep+1)  # defines the mutator's instance
    pars[HillClimbing] = {"initializer": prm_rnd_vint(0, max_rep+1), "nh_function": mutator, "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    pars[SimulatedAnnealing] = {"initializer": prm_rnd_vint(0, max_rep+1), "nh_function": mutator, "nh_size": pop_size,
                                "control": control, "update_rate": update_rate}
    # Defines GA's parameters
    p_m = 0.3
    pars[GeneticAlgorithm] = {"initializer": prm_rnd_mint(0, max_rep+1), "selector": prm_tournament(pressure=0.1),
                              "mutator": mutator, "crossover": one_point_xo, "p_m": p_m, "p_c": 1-p_m,
                              "pop_size": pop_size, "elitism": True, "reproduction": False}

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
