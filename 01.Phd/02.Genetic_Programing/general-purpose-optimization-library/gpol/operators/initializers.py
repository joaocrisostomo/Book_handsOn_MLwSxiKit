""" Initialization operators create initial candidate solutions
The module `gpol.initializers` contains some relevant initialization
operators (initializers) used to create one (for single-point solve)
or several (for population-based solve) random initial candidate
solutions in the solve space. Given the fact this library supports
different types of optimization problems (OPs), the module contains a
collection of initializers suitable for every kind of OP implemented
in this library.
"""

import copy
import math
import random
from joblib import Parallel, delayed

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from gpol.problems.inductive_programming import SML
from gpol.algorithms.genetic_algorithm import GeneticAlgorithm
from gpol.operators.selectors import prm_tournament
from gpol.operators.variators import prm_gs_mtn, prm_subtree_mtn, swap_xo
from gpol.utils.utils import rmse


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# COMBINATORIAL PROBLEMS: Knapsack
#
def prm_rnd_vint(lb=0, ub=2):
    """Generates a vector ~U{lb, hb-1}.

    Provides rnd_vint (the inner function) with the necessary outer
    scope: the lower and the upper bounds for the discrete uniform
    distribution. More precisely, lb and hb will be used as low and
    high arguments for torch.randint. For more information, consider
    PyTorch's official documentation:
    https://pytorch.org/docs/stable/generated/torch.randint.html

    Parameters
    ----------
    lb : int (default=0)
        The lower bound.
    ub : int (default=2)
        The upper bound (in fact, hb - 1).

    Returns
    -------
    rnd_vint : function
        A function which returns a vector of integers ~U{lb, hb-1}.
    """
    def rnd_vint(sspace, device="cpu"):
        """Generates a vector ~U{lb, hb-1}.

        Generates a vector of length sspace["n_dims"] ~U{lb, hb-1}. In
        practice, for the sake of operational convenience, the vector
        will be converted to torch.float. In context of this library,
        this vector represents ISA's initial solution for combinatorial
        OPs.

        Parameters
        ----------
        sspace : dict
            Problem-specific solve space (𝑆) composed of at least the
            following key-value pair:
                <"n_dims"> int: the total number of unique items in 𝑆.
        device : str (default=2)
            Specification of the processing device.

        Returns
        -------
        torch.Tensor
            A vector of length sspace["n_dims"] ~U{lb, hb-1}. For the
            sake of operational convenience, the vector will be
            converted to torch.float.
        """
        return torch.randint(low=lb, high=ub, size=(sspace["n_dims"],), device=device, dtype=torch.float)

    return rnd_vint


def prm_rnd_mint(lb=0, ub=2):
    """Generates a matrix ~U{lb, hb-1}.

    Provides rnd_mint (the inner function) with the necessary outer
    scope: the lower and the upper bounds for the discrete uniform
    distribution. More precisely, lb and hb will be used as low and
    high arguments for torch.randint. For more information, consider
    PyTorch's official documentation:
    https://pytorch.org/docs/stable/generated/torch.randint.html

    Parameters
    ----------
    lb : int (default=0)
        The lower bound.
    ub : int (default=2)
        The upper bound (in fact, hb - 1).

    Returns
    -------
    rnd_mint : function
        A function which returns a matrix of integers in [lb-hb[.
    """
    def rnd_mint(sspace, n_sols=1, device="cpu"):
        """Generates a matrix ~U{lb, hb-1}.

        Generates a matrix of with (n_sols, sspace["n_dims"])
        ~U{lb, hb-1}. In practice, for the sake of operational
        convenience, the matrix will be converted to torch.float.
        In context of this library, this matrix represents ISA's
        initial population for combinatorial OPs.

        Parameters
        ----------
        sspace : dict
            Problem-specific solve space (𝑆) composed of at least the
            following key-value pair:
                <"n_dims"> int: the total number of unique items in 𝑆.
        n_sols : int (default=1)
            The number of solutions in the population/swarm.
        device : str (default="cpu")
            Specification of the processing device.

        Returns
        -------
        torch.Tensor
            A matrix of shape (n_sols, sspace["n_dims"]) ~U{lb, hb-1}.
            For the sake of operational convenience, the matrix will be
            converted to torch.float.
        """
        return torch.randint(low=lb, high=ub, size=(n_sols, sspace["n_dims"]), device=device, dtype=torch.float)

    return rnd_mint
# ---------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# COMBINATORIAL PROBLEMS: TSP
#
def rnd_vshuffle(sspace, device="cpu"):
    """ Returns a random permutation vector

    Returns a vector of length sspace["other_cities"] that represents a
    random permutation of cities in the context of TSP.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    device : str
        Specification of the processing device.

    Returns
    -------
    torch.Tensor
        A vector of length sspace["other_cities"] that represents a
        random permutation of cities.
    """
    return sspace["other_cities"][torch.randperm(len(sspace["other_cities"]), device=device)]


def rnd_mshuffle(sspace, n_sols, device="cpu"):
    """ Returns a random permutation matrix

    Returns a matrix of shape (n_sols, sspace["other_cities"]) that
    represents a set of random permutations of cities in the context
    of TSP.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    n_sols : int
        The number of solutions in the population/swarm.
    device : str
        Specification of the processing device.

    Returns
    -------
    torch.Tensor
        A matrix of shape (n_sols, sspace["other_cities"]) that
        represents a random permutation of cities.
    """
    return torch.stack([sspace["other_cities"][torch.randperm(len(sspace["other_cities"]), device=device)] for _ in
                        range(n_sols)])
# ---------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONTINUOUS PROBLEMS
#
def rnd_vuniform(sspace, device="cpu"):
    """ Returns a uniformly distributed vector of floats

    Returns a vector of length sspace["n_dims"] generated under the
    continuous uniform distribution which parameters correspond to the
    box's constraints. In the context of this library, the output
    tensor represents the initial solution for an ISA for continuous
    problems.

    When box's constraints are regular, they are represented with a
    two-valued tensor, each value representing the lower and the upper
    bound for each of the D dimensions. When it is irregular, the
    tensor is a 2xD matrix where the 1st and the 2nd rows represent the
    lower and the upper bounds for each dimension, respectively.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    device : str
        Specification of the processing device.

    Returns
    -------
    torch.Tensor
        A vector of length sspace["n_dims"], which values are generated
        under continuous uniform distribution whose parameters
        correspond to the box's constraints.
    """
    if len(sspace["constraints"].shape) > 1:  # when the constraints is irregular
        return torch.distributions.Uniform(sspace["constraints"][0], sspace["constraints"][1]).sample().to(device)
    else:
        return torch.distributions.Uniform(sspace["constraints"][0],
                                           sspace["constraints"][1]).sample((sspace["n_dims"], )).to(device)


def rnd_muniform(sspace, n_sols, device="cpu"):
    """ Returns a uniformly distributed matrix of floats

    Returns a matrix of shape (n_solutions, sspace["n_dims"]) generated
    under continuous uniform distribution which parameters correspond
    to the hypercube's constraints. In the context of this library, the
    matrix represents ISA's initial population/swarm for continuous
    problems.

    Note that if the hypercube is regular, it is a two-valued tensor,
    each representing the lower and the upper constraints of each of the D
    dimensions of the problem; when it is intended to be irregular the
    tensor is a 2xD matrix where the 1st and the 2nd rows represent the
    lower and the upper constraints for each dimension, respectively. In the
    context of this library, the output tensor represents the initial
    solution for an ISA.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    n_sols : int
        The number of solutions in the population/swarm.
    device : str
        Specification of the processing device.

    Returns
    -------
    torch.Tensor
        A matrix of shape (n_sols, sspace["n_dims"]) with uniformly
        distributed floating-points.
    """
    if len(sspace["constraints"].shape) > 1:  # when the constraints is irregular
        return torch.distributions.Uniform(sspace["constraints"][0], sspace["constraints"][1]).sample((n_sols,)).to(device)
    else:
        return torch.distributions.Uniform(sspace["constraints"][0],
                                           sspace["constraints"][1]).sample((n_sols, sspace["n_dims"])).to(device)

    torch.distributions.Uniform(sspace["constraints"][0], sspace["constraints"][1]).sample((n_sols, sspace["n_dims"])).to(device)
# ---------------------------


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# INDUCTIVE PROGRAMMING PROBLEMS
#
def grow(sspace, device="cpu"):
    """ Implements Grow initialization algorithm for GP

    The implementation assumes the probability of sampling a program
    element from the set of functions is the same as from the set of
    terminals until achieving the maximum depth (i.e., 50%). The
    probability of selecting a constant from the set of terminals
    is controlled by the value in "p_constants" key, defined in sspace
    dictionary, and equals 0.5*sspace["p_constants"].

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.
    device : str
        Specification of the processing device.

    Returns
    -------
    program : list
        A list of program elements which represents an initial computer
        program (candidate solution). The program follows LISP-based
        formulation and Polish pre-fix notation.
    """
    function_ = random.choice(sspace["function_set"])  # starts the tree with a function
    program = [function_]
    terminal_stack = [function_.arity]
    max_depth = random.randint(1, sspace["max_init_depth"])

    while terminal_stack:
        depth = len(terminal_stack)
        choice = random.randint(0, 1)  # 0: function_, 1: terminal

        if (depth < max_depth) and choice == 0:
            function_ = random.choice(sspace["function_set"])
            program.append(function_)
            terminal_stack.append(function_.arity)
        else:
            if random.uniform(0, 1) < sspace["p_constants"]:
                terminal = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)]
            else:
                terminal = random.randint(0, sspace["n_dims"] - 1)

            program.append(terminal)
            terminal_stack[-1] -= 1
            while terminal_stack[-1] == 0:
                terminal_stack.pop()
                if not terminal_stack:
                    return program
                terminal_stack[-1] -= 1
    return None


def prm_grow(sspace):
    """ Implements Grow initialization algorithm

    The library's interface restricts variation operators' parameters
    to solutions' representations only. However, the functioning of
    some of the GP's variation operators requires random trees'
    generation - this is the case of the sub-tree mutation, the
    geometric semantic operators, ... In this sense, the variation
    functions' enclosing scope does not contain enough information to
    generate the initial trees. To remedy this situation, closures are
    used as they provide the variation functions the necessary outer
    scope for trees' initialization: the solve space. Moreover, this
    solution, allows one to have a deeper control over the operators'
    functioning - an important feature for the research purposes.

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    grow_ : function
        A function which implements Grow initialization algorithm,
        which uses the user-specified solve space for trees'
        initialization.
    """
    def grow_():
        """ Implements Grow initialization algorithm

        Implements Grow initialization algorithm, which uses the user-
        specified solve space for trees' initialization.

        Returns
        -------
        program : list
            A list of program elements which represents an initial computer
            program (candidate solution). The program follows LISP-based
            formulation and Polish pre-fix notation.
        """
        function_ = random.choice(sspace["function_set"])
        program = [function_]
        terminal_stack = [function_.arity]
        max_depth = random.randint(1, sspace["max_init_depth"])

        while terminal_stack:
            depth = len(terminal_stack)
            choice = random.randint(0, 1)  # 0: function_, 1: terminal

            if (depth < max_depth) and choice == 0:
                function_ = random.choice(sspace["function_set"])
                program.append(function_)
                terminal_stack.append(function_.arity)
            else:
                if random.uniform(0, 1) < sspace["p_constants"]:
                    terminal = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)]
                else:
                    terminal = random.randint(0, sspace["n_dims"] - 1)

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return None

    return grow_


def full(sspace, device="cpu"):
    """ Implements Full initialization algorithm

    The probability of selecting a constant from the set of terminals
    is controlled by the value in "p_constants" key, defined in sspace
    dictionary, and equals 0.5*sspace["p_constants"].

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.
    device : str
        Specification of the processing device.

    Returns
    -------
    program : list
        A list of program elements which represents an initial computer
        program (candidate solution). The program follows LISP-based
        formulation and Polish pre-fix notation.
    """
    function_ = random.choice(sspace["function_set"])
    program = [function_]
    terminal_stack = [function_.arity]

    while terminal_stack:
        depth = len(terminal_stack)

        if depth < sspace["max_init_depth"]:
            function_ = random.choice(sspace["function_set"])
            program.append(function_)
            terminal_stack.append(function_.arity)
        else:
            if random.uniform(0, 1) < sspace["p_constants"]:
                terminal = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)]
            else:
                terminal = random.randint(0, sspace["n_dims"] - 1)

            program.append(terminal)
            terminal_stack[-1] -= 1
            while terminal_stack[-1] == 0:
                terminal_stack.pop()
                if not terminal_stack:
                    return program
                terminal_stack[-1] -= 1
    return None


def prm_full(sspace):
    """ Implements Full initialization algorithm

    The library's interface restricts variation operators' parameters
    to solutions' representations only. However, the functioning of
    some of the GP's variation operators requires random trees'
    generation - this is the case of the sub-tree mutation, the
    geometric semantic operators, ... In this sense, the variation
    functions' enclosing scope does not contain enough information to
    generate the initial trees. To remedy this situation, closures are
    used as they provide the variation functions the necessary outer
    scope for trees' initialization: the solve space. Moreover, this
    solution, allows one to have a deeper control over the operators'
    functioning - an important feature for the research purposes.

    Parameters
    ----------
    sspace : dict
        Problem instance's solve-space.

    Returns
    -------
    full_ : function
        A function which implements Full initialization algorithm,
        which uses the user-specified solve space for trees'
        initialization.
    """
    def full_():
        """ Implements Full initialization algorithm

        Implements Full initialization algorithm, which uses the user-
        specified solve space for trees' initialization.

        Returns
        -------
        program : list
            A list of program elements which represents an initial computer
            program (candidate solution). The program follows LISP-based
            formulation and Polish pre-fix notation.
        """
        function_ = random.choice(sspace["function_set"])
        program = [function_]
        terminal_stack = [function_.arity]

        while terminal_stack:
            depth = len(terminal_stack)

            if depth < sspace["max_init_depth"]:
                function_ = random.choice(sspace["function_set"])
                program.append(function_)
                terminal_stack.append(function_.arity)
            else:
                if random.uniform(0, 1) < sspace["p_constants"]:
                    terminal = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)]
                else:
                    terminal = random.randint(0, sspace["n_dims"] - 1)

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return None

    return full_


def rhh(sspace, n_sols, device="cpu"):
    """ Implements Ramped Half and Half initialization algorithm

    Implements the Ramped Half and Half, which, by itself, uses
    Full and Grow.

    Parameters
    ----------
    sspace : dict
        Problem-specific solve space (𝑆).
    n_sols : int
        The number of solutions in the population
    device : str
        Specification of the processing device.

    Returns
    -------
    pop : list
        A list of program elements which represents the population
        initial of computer programs (candidate solutions). Each
        program is a list of program's elements that follows a
        LISP-based formulation and Polish pre-fix notation.
    """
    pop = []
    n_groups = sspace["max_init_depth"]
    group_size = math.floor((n_sols / 2.) / n_groups)

    for group in range(n_groups):
        max_depth_group = group + 1
        for i in range(group_size):
            sspace_group = sspace
            sspace_group["max_init_depth"] = max_depth_group
            pop.append(full(sspace_group, device))
            pop.append(grow(sspace_group, device))

    while len(pop) < n_sols:
        pop.append(grow(sspace_group, device) if random.randint(0, 1) else full(sspace_group, device))
    return pop


def prm_edda(X, y, train_indices, test_indices=None, deme_size=100, maturation=5, p_gp=0.5, p_features=0.5,
             p_samples=0.5, p_functions=0.5, replacement=True, bnd_pressure=(0.05, 0.2), ms=torch.tensor([1.0]),
             batch_size=50, shuffle=True, ffunction=rmse, min_=True, verbose=0, log=0, n_jobs=1):
    """Evolutionary Demes Despeciation Algorithm.

    EDDA is an initialization technique inspired by the biological
    phenomenon of demes despeciation, i.e. the combination of demes
    of previously distinct species into a new population. In synthesis,
    the initial population of GSGP programming is created using the
    best individuals obtained from a set of separate sub-populations,
    or demes, evolved for few generations under distinct evolutionary
    conditions: some of which run standard GP operators and the others
    geometric semantic operators. GSGP, with this initialization
    technique, is shown to outperform GSGP using the traditional RHH.
    More specifically, it was empirically demonstrated that, on the
    studied problems, the proposed initialization technique allows to
    generate solutions with comparable or even better generalization
    ability, and of significantly smaller size than the RHH.

    This implementation assumes prm_gs_mtn, where full representation
    of the individuals is held in memory, and prm_tournament where the
    tournament pool is randomly drawn for each deme.

    References
    ----------
    Leonardo Vanneschi, Illya Bakurov and Mauro Castelli.
        "An initialization technique for geometric semantic GP based
        on demes evolution and despeciation". IEEE Congress on
        Evolutionary Computation. 2017.
    I. Bakurov, L. Vanneschi, M. Castelli and F. Fontanella.
        "EDDA-V2: an improvement of the evolutionary demes despeciation
        algorithm". Parallel Problem Solving from Nature – PPSN XV.
        2018.

    Parameters
    ----------
    X : torch.Tensor
        A tensor representing the input samples. Internally, it will be
        converted to dtype=torch.float. It is assumed that the object
        was already allocated on the proper processing device.
    y : torch.Tensor
        A tensor representing the The target values. Internally, it
        will be converted to dtype=torch.float. It is assumed that the
        object was already allocated on the proper processing device.
    train_indices : torch.Tensor
        Indices representing the training partition of the data.
    test_indices : torch.Tensor (default=None)
        Indices representing the testing partition of the data.
    deme_size : int (default=100)
        Size of each deme (i.e., sub-population).
    maturation : int (default=5)
        The number of generations each deme is evolved.
    p_gp : float (default=0.5)
        Proportion of GP demes in EDDA.
    p_features: float (default=0.5)
        The proportion of features to randomly draw from the features'
        set to compose the solve-space of each deme.
    p_samples: float (default=0.5)
        The proportion of samples to draw from X to evolve each deme.
        The samples are drawn randomly, with (bootstrap samples) or
        without replacement.
    p_functions: float (default=0.5)
        The proportion of primitive functions to draw from the
        functions' set to compose the solve-space of each deme.
    replacement : bool (default=True)
        Whether data samples are drawn with replacement (bootstrap
        samples, use True) or not (use False).
    bnd_pressure : tuple
        The lower and the upper bound for the tournament's selection
        pressure. The value will be randomly drawn at the beginning of
        demes' evolution ~U(pressure_constraints[0], pressure_constraints[1]).
    ms : torch.Tensor (default torch.tensor([1.0], device="cpu")
        The mutation step, provided as a single or a 1D tensor
        of length m. When it is a single tensor, the mutation
        step equals ms; if it is a 1D tensor of length m, then
        the mutation step is selected from it at random, at
        each application of the operator.
    batch_size : int (default=50)
        Batch size.
    shuffle : bool (default=True)
        Whether to shuffle the data before selecting the batch.
        DataLoader for the testing set.
    ffunction : function (default=rmse)
        𝑓 : 𝑆 → 𝐼𝑅. This allows the user to apply a fitness function
        during demes' evolution different from the main evolutionary
        process.
    min_ : bool (default=True)
        The optimization purpose of the user-specified fitness function.
    n_jobs : int (default=1)
        The number of parallel processes used to evolve the demes.
    verbose : int, optional (default=0)
        An integer that controls the verbosity of the solve process.
        The following nomenclature is applied in this class:
            - verbose = 0: do not print anything.
            - verbose = 1: prints current iteration, its timing, and
             elite's length and fitness.
            - verbose = 2: also prints population's average and
             standard deviation (in terms of fitness).
    log : int, optional (default=0)
            An integer that controls the completeness of the data that
            is written in the log file. The following nomenclature is
            applied in this class:
                - log = 0: do not write any log data.
                - log = 1: writes the current iteration, its timing,
                 and elite's length and fitness.
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness).
                - log = 3: also, writes elite's representation.
    n_jobs: int (default=1)
        Number of parallel processes used to evolve demes.

    Returns
    -------
    edda : function
        A function which implements EDDA [1].
    """
    def edda(sspace, n_sols, device="cpu"):
        """ Evolutionary Demes Despeciation Algorithm.

        Parameters
        ----------
        sspace : dict
            Problem's solve-space with the following key-value pairs:
            <"n_dims"> int: number of input features of a given SML
                problem (a.k.a. input dimensions or dimensionality).
            <"function_set"> list: a list of primitive functions, i.e.
                gpol.utils.inductive_programming._Function objects.
            <"constant_set"> torch.Tensor: a tensor holding constant
                which function as additional terminals in a GP tree.
            <"p_constants"> float: the probability of selecting a
                constant, instead of an input feature, as a terminal
                in a GP tree.
            <"max_init_depth"> int: maximum allowed depth during trees'
                initialization. If EDDA initialization is used, this
                parameter regards to demes' initial trees.
            <"max_depth"> int: maximum allowed trees' depth during the
                evolution. This parameter is automatically ignored if
                GSGP algorithm is used.
            <"n_batches"> int: number of batches used to assess the
                candidate solution at a given iteration. This parameter
                is automatically ignored if GSGP algorithm is used.
        n_sols : int
            Population's size. Corresponds to the number of independent
            demes.
        device : str (default="cpu)
            Specification of the processing device.

        Returns
        -------
        pop : list
            A list of program elements which represents the population
            initial of computer programs (candidate solutions). Each
            program is a list of program's elements that follows a
            LISP-based formulation and Polish pre-fix notation.
        """
        # Creates a flag to whether to test the elite or not
        test_elite = True if test_indices is not None else False
        # Computes the number of GP and GSGP demes
        n_gp_demes = int(p_gp * n_sols)
        # Computes the number of features to sample, if necessary
        if p_features and p_features < 1.0:
            n_dims = int(X.shape[1]*p_features)
            features_map = {}

        # Performs train/test split
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = (X[test_indices], y[test_indices]) if test_indices is not None else (None, None)

        # Adjusts the batch size
        batch_size_ = len(X_train) if batch_size is None else batch_size

        # Creates an instance of RandomSampler for data instances' sampling
        num_samples = int(len(X_train)*p_samples) if (p_samples is not None and replacement) else None
        rs = RandomSampler(X_train, replacement=replacement, num_samples=num_samples)

        # Utility function to perform parallel demes' evolution
        def _parallel_evolve_deme(i, gp_deme, seed_):
            # Sets the random state
            torch.manual_seed(seed_)
            random.seed(seed_)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # > > > Creates the solve space for the deme i
            # Copies the solve space
            sspace_i = copy.deepcopy(sspace)

            # Updates the functions' set, if necessary
            if p_functions and p_functions < 1.0:
                sspace_i["function_set"] = random.sample(sspace["function_set"],
                                                         int(len(sspace["function_set"]) * p_functions))

            # Takes a bootstrap sample
            bootstrap_idxs = list(rs)
            X_train_i, y_train_i = X_train[bootstrap_idxs, :], y_train[bootstrap_idxs]

            # Performs features' sampling
            if p_features and p_features < 1.0:
                sspace_i["n_dims"] = n_dims
                features_i = random.sample(list(range(sspace["n_dims"])), n_dims)
                X_train_i = X_train_i[:, features_i]
                X_test_i = X_test[:, features_i]
            else:
                X_test_i = X_test
                features_i = None

            # Creates training and test dataloaders
            dl_train = DataLoader(dataset=TensorDataset(X_train_i, y_train_i), batch_size=batch_size_, shuffle=shuffle)
            dl_test = DataLoader(dataset=TensorDataset(X_test_i, y_test), batch_size=batch_size_, shuffle=shuffle)

            # Creates a problem instance (PI)
            pi = SML(sspace_i, ffunction=ffunction, dl_train=dl_train, dl_test=dl_test, min_=min_)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # > > > Defines algorithms' parameters
            # Creates single trees' initializer
            sp_init = prm_full(sspace_i)
            # Randomly chooses the tournament selection's pressure
            pressure = random.uniform(bnd_pressure[0], bnd_pressure[1])

            # Chooses the variation operators and the respective probabilities
            if gp_deme:
                isa_name = "GP"
                p_m = random.uniform(0, 1)
                mutator, crossover = prm_subtree_mtn(sp_init), swap_xo
            else:
                isa_name = "GSGP"
                p_m = 1.0
                mutator, crossover = prm_gs_mtn(sp_init, ms), None

            # Creates an instance of the solve algorithm
            isa = GeneticAlgorithm(pi=pi, initializer=rhh, selector=prm_tournament(pressure=pressure), mutator=mutator,
                                   crossover=crossover, p_m=p_m, p_c=1.0-p_m,  pop_size=deme_size, elitism=True,
                                   reproduction=False, seed=seed_, device=device)
            isa.__name__ = "EDDA-" + isa_name

            if verbose > 0:
                print("EDDA: deme Nº {0}, deme type {1}, seed {2}".format(i, isa_name, seed_))

            # Solves the PI
            isa.solve(n_iter=maturation, start_at=None, test_elite=test_elite, verbose=verbose, log=log)
            return {i: (isa.best_sol.repr_, features_i)}

        # Performs evolution and despeciation of GSGP demes
        seed_ = random.randint(0, 1000000)
        pool = Parallel(n_jobs=n_jobs)(delayed(_parallel_evolve_deme)(i, True, i + seed_) for i in range(n_gp_demes))
        if n_gp_demes != n_sols:
            pool.extend(Parallel(n_jobs=n_jobs)(delayed(_parallel_evolve_deme)(i, False, i+seed_) for i in range(n_gp_demes, n_sols)))
        pool = dict((key, value) for d in pool for key, value in d.items())

        pop = []
        for i in range(n_sols):
            pop.append(pool[i][0])
            if p_features and p_features < 1.0:
                features_map[i] = pool[i][1]

        # Overrides features' names, if features' sub-sampling was used
        if p_features and p_features < 1.0:
            for i, repr_ in enumerate(pop):
                features_map_i = dict(zip(list(range(n_dims)), features_map[i]))
                for pe_i in range(len(repr_)):
                    if isinstance(repr_[pe_i], int):
                        repr_[pe_i] = features_map_i[repr_[pe_i]]

        return pop

    return edda
