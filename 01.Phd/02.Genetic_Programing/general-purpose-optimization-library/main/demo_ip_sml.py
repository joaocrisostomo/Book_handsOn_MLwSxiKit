def main():
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    from gpol.problems.inductive_programming import SML
    from gpol.utils.datasets import load_boston
    from gpol.algorithms.random_search import RandomSearch
    from gpol.algorithms.local_search import HillClimbing, SimulatedAnnealing
    from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
    from gpol.operators.initializers import rhh, prm_grow, grow
    from gpol.operators.selectors import prm_tournament
    from gpol.operators.variators import prm_subtree_mtn, swap_xo
    from gpol.utils.utils import train_test_split, rmse
    from gpol.utils.inductive_programming import function_map

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > > Demonstration of how to create an instance of SML-IPOP
    # Defines the processing device and random state 's seed
    device, seed, p_test = 'cpu', 0, 0.3
    # Creates training and test DataLoader
    batch_size, shuffle, num_workers = 50, True, 0
    # Imports the data
    X, y = load_boston(X_y=True)
    # Performs train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=p_test, seed=seed)
    # Creates two objects of type TensorDataset: one for training, another for test data
    ds_train = TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # Characterizes the program elements: function set
    function_set = [function_map["add"], function_map["sub"], function_map["mul"], function_map["div"]]
    # Characterizes the program elements: constant set
    constant_set = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)
    # Defines the solve space
    sspace = {"n_dims": X.shape[1], "function_set": function_set, "constant_set": constant_set, "p_constants": 0.1,
              "max_init_depth": 5, "max_depth": 15, "n_batches": 1}
    # Creates problem's instance
    pi = SML(sspace, rmse, dl_train, dl_test, min_=True)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > >´Defines algorithms' parameters
    pop_size, n_iter = 100, 30
    # Defines RS's parameters
    pars = {RandomSearch: {"initializer": grow}}
    # Defines HC's parameters
    single_initializer = prm_grow(sspace)
    pars[HillClimbing] = {"initializer": grow, "nh_function": prm_subtree_mtn(single_initializer), "nh_size": pop_size}
    # Defines SA's parameters
    control, update_rate = 1, 0.9
    pars[SimulatedAnnealing] = {"initializer": grow, "nh_function": prm_subtree_mtn(single_initializer),
                                "nh_size": pop_size, "control": control, "update_rate": update_rate}
    # Defines GP's parameters
    p_m = 0.3
    pars[GeneticAlgorithm] = {"initializer": rhh, "selector": prm_tournament(0.1),
                              "mutator": prm_subtree_mtn(single_initializer), "crossover": swap_xo, "p_m": p_m,
                              "p_c": 1.0-p_m, "pop_size": pop_size, "elitism": True, "reproduction": True}

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # > > >´Shows how to use the algorithms to solve the PI
    for isa_type, isa_pars in pars.items():
        # Creates an instance of a solve algorithm
        isa = isa_type(pi=pi, **isa_pars, seed=seed, device=device)
        # n_iter*pop_size if isinstance(isa, RandomSearch) else n_iter
        # Solves the PI
        isa.solve(n_iter=n_iter, tol=0.5, n_iter_tol=5, test_elite=True, verbose=2, log=0)
        print("Algorithm: {}".format(isa_type.__name__))
        print("Best solution's fitness: {:.3f}".format(isa.best_sol.fit))
        print("Best solution:", isa.best_sol.repr_)


if __name__ == "__main__":
    main()

