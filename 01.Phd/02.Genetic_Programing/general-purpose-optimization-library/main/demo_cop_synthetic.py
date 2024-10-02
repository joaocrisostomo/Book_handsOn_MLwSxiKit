""" Examples
A simple demonstration of how to solve a Continuous Optimization Problem
(COP) using GPOL's algorithms that are appropriate for this kind of
optimization problem. This example served as a basis for the paper.
"""


def main():
    import torch
    from gpol.problems.continuous import Box
    from gpol.problems.utils import rastrigin_function
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing, SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.algorithms.differential_evolution import DifferentialEvolution
    from gpol.algorithms.swarm_intelligence import SPSO, APSO
    from gpol.operators.initializers import rnd_muniform, rnd_vuniform
    from gpol.operators.selectors import prm_dernd_selection, prm_tournament
    from gpol.operators.variators import prm_pso, prm_iball_mtn, geometric_xo, de_rand, de_exponential_xo

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to create an instance of Box COP
    # Defines the processing device and the random state 's seed
    device, seed = "cpu", 0
    # Creates the solve space
    sspace = {"n_dims": 2, "constraints": torch.tensor([-5.12, 5.12], device=device)}
    # Creates an instance of optimization problem (PI)
    pi = Box(sspace=sspace, ffunction=rastrigin_function, min_=True, bound=True)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Defines algorithms' parameters
    # Defines the computational resources
    pop_size, n_iter = 100, 30
    # Defines RS's parameters
    pars = {RandomSearch: {"initializer": rnd_vuniform}}
    # Defines HC's parameters
    p_m_ball, radius = 1.0, 0.3  # probability of applying ball mutation at a given index with a given radius
    mutator = prm_iball_mtn(p_m_ball, radius)
    pars[HillClimbing] = {"initializer": rnd_vuniform, "nh_function": mutator, "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    pars[SimulatedAnnealing] = {"initializer": rnd_vuniform, "nh_function": mutator, "nh_size": pop_size,
                                "control": control, "update_rate": update_rate}
    # Defines GA's parameters
    p_m = 0.5
    pars[GeneticAlgorithm] = {"initializer": rnd_muniform, "selector": prm_tournament(0.1), "mutator": mutator,
                              "crossover": geometric_xo, "p_m": p_m, "p_c": 1.0 - p_m, "pop_size": pop_size,
                              "elitism": True, "reproduction": False}
    # Defines PSO's parameters
    social_f, cognitive_f, w_max, w_min, v_clamp = 2.0, 2.0, 0.9, 0.4, 3.0
    pso_update_rule = prm_pso(c1=social_f, c2=cognitive_f, w_max=w_max, w_min=w_min)
    pars[SPSO] = {"initializer": rnd_muniform, "mutator": pso_update_rule, "v_clamp": v_clamp, "pop_size": pop_size}
    pars[APSO] = {"initializer": rnd_muniform, "mutator": pso_update_rule, "v_clamp": v_clamp, "pop_size": pop_size}
    # Defines DE's parameters
    n_parents, m_weights = 3, torch.tensor([0.9], device=device)
    pars[DifferentialEvolution] = {"initializer": rnd_muniform, "selector": prm_dernd_selection(n_sols=3),
                                   "mutator": de_rand, "crossover": de_exponential_xo, "m_weights": m_weights,
                                   "c_rate": 0.5, "pop_size": pop_size}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Iterates algorithms and their respective parameters
    for isa_type, isa_pars in pars.items():
        # Creates an instance of a solve algorithm
        isa = isa_type(pi=pi, **isa_pars, seed=seed, device=device)
        # n_iter*pop_size if isinstance(isa, RandomSearch) else n_iter
        # Solves the PI
        isa.solve(n_iter=n_iter, tol=0.2, n_iter_tol=5, verbose=2, log=0)
        print("Algorithm: {}".format(isa_type.__name__))
        print("Best solution's fitness: {:.3f}".format(isa.best_sol.fit))
        print("Best solution:", isa.best_sol.repr_)


if __name__ == "__main__":
    main()
