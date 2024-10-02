import os
import time
import random
import pickle
import logging

import torch
import pandas as pd

from gpol.algorithms.population_based import PopulationBased
from gpol.utils.population import Population
from gpol.utils.inductive_programming import _execute_tree, _get_tree_depth


class GeneticAlgorithm(PopulationBased):
    """Implements Genetic Algorithm (GA).

    Genetic Algorithm (GA) is a meta-heuristic introduced by John
    Holland, strongly inspired by Darwin's Theory of Evolution.
    Conceptually, the algorithm starts with a random-like population of
    candidate-solutions (called chromosomes). Then, by resembling the
    principles of natural selection and the genetically inspired
    variation operators, such as crossover and mutation, the algorithm
    breeds a population of next-generation candidate-solutions (called
    the offspring population), which replaces the previous population
    (a.k.a. the population of parents). The procedure is iterated until
    reaching some stopping criteria, like a predefined number of
    iterations (also called generations).

    An instance of GA can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) a function to initialize the solve at a given point in 𝑆;
        3) a function to select candidate solutions for variation phase;
        4) a function to mutate candidate solutions;
        5) the probability of applying mutation;
        6) a function to crossover two solutions (the parents);
        7) the probability of applying crossover;
        8) the population's size;
        9) the best solution found by the PB-ISA;
        10) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased))
        The best solution found.
    pop_size : int (inherited from PopulationBased)
        The population's size.
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from PopulationBased))
        The initialization procedure.
    selector : function
        The selection procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    p_m : float
        The probability of applying mutation.
    crossover : function
        The crossover procedure.
    p_c : float
        The probability of applying crossover.
    elitism : bool
        A flag which activates elitism during the evolutionary process.
    reproduction : bool
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "GeneticAlgorithm"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.2, p_c=0.8, pop_size=100, elitism=True,
                 reproduction=False, seed=0, device="cpu"):
        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        selector : function
            The selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            The crossover function.
        p_m : float (default=0.2)
            Probability of applying mutation.
        p_c : float (default=0.8)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.p_c = p_c
        self.elitism = elitism
        self.reproduction = reproduction

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a GA.

        This method implements the following pseudo-code:
            1) Create a random initial population of size n (𝑃);
            2) Repeat until satisfying some termination condition,
             typically the number of generations:
                1) Calculate fitness ∀ individual in 𝑃;
                2) Create an empty population 𝑃’, the population of
                 offsprings;
                3) Repeat until 𝑃’ contains 𝑛 individuals:
                    1) Chose the main genetic operator – crossover,
                     with probability p_c or reproduction with
                     probability (1 − p_c);
                    2) Select two individuals, the parents, by means
                     of a selection algorithm;
                    3) Apply operator selected in 2) 3) 1) to the
                     individuals selected in 2) 3) 2);
                    4) Apply mutation on the resulting offspring with
                     probability p_m;
                    5) Insert individuals from 2) 3) 4) into 𝑃’;
                4) Replace 𝑃 with 𝑃’;
            3) Return the best individual in 𝑃 (the elite).

        Parameters
        ----------
        n_iter : int (default=20)
            The number of iterations.
        tol : float (default=None)
            Minimum required fitness improvement for n_iter_tol
            consecutive iterations to continue the solve. When best
            solution's fitness is not improving by at least tol for
            n_iter_tol consecutive iterations, the solve will be
            automatically interrupted.
        n_iter_tol : int (default=5)
            Maximum number of iterations to continue the solve while
            not meeting the tol improvement.
        start_at : object (default=None)
            The initial starting point in 𝑆 (it is is assumed to be
            feasible under 𝑆's constraints, if any).
        test_elite : bool (default=False)
            Indicates whether assess the best-so-far solution on the
            test partition (this regards SML-based OPs).
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything.
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness.
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness);
                - log = 3: also, writes elite's representation.
        """
        # Optionally, tracks initialization's timing for console's output
        if verbose > 0:
            start = time.time()

        # 1)
        self._initialize(start_at=start_at)
        # Optionally, evaluates the elite on the test partition
        if test_elite:
            self.pi.evaluate_sol(self.best_sol, test=test_elite)

        # Optionally, computes population's AVG and STD (in terms of fitness)
        if log == 2 or verbose == 2:
            self.pop.fit_avg = self.pop.fit.mean().item()
            self.pop.fit_std = self.pop.fit.std().item()

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, self.pop, verbose)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, self.pop, log)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        for it in range(1, n_iter + 1, 1):
            # 2) 2)
            offs_pop, start = [], time.time()

            # 2) 3)
            pop_size = self.pop_size - self.pop_size % 2
            while len(offs_pop) < pop_size:
                # 2) 3) 2)
                p1_idx = p2_idx = self.selector(self.pop, self.pi.min_)
                # Avoids selecting the same parent twice
                while p1_idx == p2_idx:
                    p2_idx = self.selector(self.pop, self.pi.min_)

                if not self.reproduction:  # performs GP-like variation
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    else:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])
                else:  # performs GA-like variation
                    offs1, offs2 = self.pop[p1_idx], self.pop[p2_idx]
                    if random.uniform(0, 1) < self.p_c:
                        # 2) 3) 3)
                        offs1, offs2 = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    if random.uniform(0, 1) < self.p_m:
                        # 2) 3) 4)
                        offs1 = self.mutator(self.pop[p1_idx])
                        offs2 = self.mutator(self.pop[p2_idx])

                # 2) 3) 5)
                offs_pop.extend([offs1, offs2])

            # Adds one more individual, if the population size is odd
            if pop_size < self.pop_size:
                offs_pop.append(self.mutator(self.pop[self.selector(self.pop, self.pi.min_)]))

            # If batch training, appends the elite to evaluate_pop it on the same batch(es) as the offspring population
            if self._batch_training:
                offs_pop.append(self.best_sol.repr_)

            # If solutions are objects of type torch.Tensor, stacks their representations in the same tensor
            if isinstance(offs_pop[0], torch.Tensor):
                offs_pop = torch.stack(offs_pop)

            # 2) 1)
            offs_pop = Population(offs_pop)
            self.pi.evaluate_pop(offs_pop)

            # Overrides elites's information, if it was re-evaluated, and removes it from 'offsprings'
            if self._batch_training:
                self.best_sol.valid = offs_pop.valid[-1]
                self.best_sol.fit = offs_pop.fit[-1]
                # Removes the elite from the object 'offsprings'
                offs_pop.repr_ = offs_pop.repr_[0:-1]
                offs_pop.valid = offs_pop.valid[0: -1]
                offs_pop.fit = offs_pop.fit[0: -1]

            # Updates the current elite
            best_offs = self._get_best_pop(offs_pop)
            self.best_sol = self._get_best(self.best_sol, best_offs)

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=test_elite)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log == 2 or verbose == 2:
                offs_pop.fit_avg = offs_pop.fit.mean().item()
                offs_pop.fit_std = offs_pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, offs_pop, log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, offs_pop, verbose)

            # Performs population's replacement
            if self.elitism:
                self._elite_replacement(offs_pop, best_offs)
            else:
                self.pop = offs_pop

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

    def _elite_replacement(self, offs_pop, best_offs):
        """Replaces 𝑃 with 𝑃’ and preserves the elite.

        This method directly replaces 𝑃 with 𝑃’ if 𝑃’ already contains
        the elite, i.e., the elite is the best offspring which is
        already in 𝑃’. Otherwise, when the elite is the best parent,
        𝑃 is replaced with 𝑃’ and randomly selected offspring is
        replaced with it.

        Parameters
        ----------
        offs_pop : Population
            The population of offsprings, 𝑃’.
        best_offs : Solution
            The elite of 𝑃’.
        """
        if self.best_sol == best_offs:
            # Directly overrides 𝑃 with 𝑃’ if 𝑃’ already contains the elite
            self.pop = offs_pop
        else:
            # Inserts 'best_sol', the best parent, in 'offs_pop' at a random index
            index = random.randint(0, self.pop_size - 1)
            repr_ = self.best_sol.repr_.copy() if type(self.best_sol.repr_) == list else self.best_sol.repr_.clone()
            offs_pop.repr_[index] = repr_
            offs_pop.fit[index] = self.best_sol.fit.clone()
            if hasattr(self.best_sol, "valid"):
                offs_pop.valid[index] = self.best_sol.valid

            self.pop = offs_pop


class GSGP(GeneticAlgorithm):
    """Re-implements Genetic Algorithm (GA) for GSGP.

    Given the growing importance of the Geometric Semantic Operators
    (GSOs), proposed by Moraglio et al., we decided to include them in
    our library, following the efficient implementation proposed by
    Castelli et al. More specifically, we implemented Geometric
    Semantic Genetic Programming (GSGP) through a specialized class
    called GSGP, subclass of the GeneticAlgorithm, that encapsulates
    the efficient implementation of GSOs.

    An instance of GSGP can be characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) the population's size;
        3) a function to initialize the solve at a given point in 𝑆;
        4) a function to select candidate solutions for variation phase;
        5) a function to mutate candidate solutions;
        6) the probability of applying mutation;
        7) a function to crossover two solutions (the parents);
        8) the probability of applying crossover;
        9) whether to apply elite replacement or not;
        10) whether to apply reproduction or not;
        11) the best solution found by the PB-ISA;
        12) a collection of candidate solutions - the population;
        11) a random state for random numbers generation;
        12) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from GeneticAlgorithm)
        An instance of OP.
    best_sol : Solution (inherited from GeneticAlgorithm))
        The best solution found.
    pop_size : int (inherited from GeneticAlgorithm)
        The population's size.
    pop : Population (inherited from GeneticAlgorithm)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from GeneticAlgorithm)
        The initialization procedure.
    selector : function (inherited from GeneticAlgorithm)
        The selection procedure.
    mutator : function (inherited)
        The mutation procedure.
    crossover : function (inherited from GeneticAlgorithm)
        The crossover procedure.
    p_m : float (inherited from GeneticAlgorithm)
        The probability of applying mutation.
    p_c : float (inherited from GeneticAlgorithm)
        The probability of applying crossover.
    elitism : bool (inherited from GeneticAlgorithm)
        A flag which activates elitism during the evolutionary process.
    reproduction : bool (inherited from GeneticAlgorithm)
        A flag which states if reproduction should happen (reproduction
        is True), when the crossover is not applied. If reproduction is
        False, then either crossover or mutation will be applied.
        path_init_pop : str
        Connection string towards initial population's repository.
    path_init_pop : str
        Connection string towards initial trees' repository.
    path_rts : str
        Connection string towards random trees' repository.
    history : dict
        Dictionary which stores the history of operations applied on
        each offspring. In abstract terms, it stores 1 level family
        tree of a given offspring. More specifically, history stores
        as a key the offspring's ID, as a value a dictionary with the
        following structure:
            - "Iter": iteration's number;
            - "Operator": the variation operator that was applied on
             a given offspring;
            - "T1": the ID of the first parent;
            - "T2": the ID of the second parent (if GSC was applied);
            - "Tr": the ID of a random tree generated (assumes only
             one random tree is necessary to apply an operator);
            - "ms": mutation's step (if GSM was applied);
            - "Fitness": offspring's training fitness;
    pop_ids : lst
        IDs of the current population (the population of parents).
    seed : int (inherited from GeneticAlgorithm)
        The seed for random numbers generators.
    device : str (inherited from GeneticAlgorithm)
        Specification of the processing device.

    References
    ----------
    Alberto Moraglio, Krzysztof Krawiec and Colin G. Johnson.
        "Geometric Semantic Genetic Programming". Parallel Problem
        Solving from Nature - PPSN XII. 2012
    Mauro Castelli, Sara Silva and Leonardo Vanneschi
        "A C++ framework for geometric semantic genetic programming".
        Genetic Programming and Evolvable Machines. 2015.
    """
    __name__ = "GSGP"

    def __init__(self, pi, initializer, selector, mutator, crossover, p_m=0.95, p_c=0.05, pop_size=100, elitism=True,
                 reproduction=False, path_init_pop=None, path_rts=None, seed=0, device="cpu"):

        """ Objects' constructor

        Following the main purpose of a PB-ISA, the constructor takes a
        problem instance (PI) to solve, the population's size and an
        initialization procedure. Moreover it receives the mutation and
        the crossover functions along with the respective probabilities.
        The constructor also takes two boolean values indicating whether
        elitism and reproduction should be applied. Finally, it takes
        some technical parameters like the random seed and the processing
        device.

        Parameters
        ----------
        pi : Problem
            Optimization problem's instance (PI).
        path_init_pop : str
            Connection string towards initial trees' repository.
        path_rts : str
            Connection string towards random trees' repository.
        initializer : function
            The initialization procedure.
        selector : function
            Selection procedure.
        mutator : function
            A function to move solutions across the solve space.
        crossover : function
            Crossover.
        p_m : float (default=0.05)
            Probability of applying mutation.
        p_c : float (default=0.95)
            Probability of applying crossover.
        pop_size : int (default=100)
            Population's size.
        elitism : bool (default=True)
            A flag which activates elitism during the evolutionary process.
        reproduction : bool (default=False)
            A flag which states if reproduction should happen (reproduction
            is True), when the crossover is not applied.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        GeneticAlgorithm.__init__(self, pi, initializer, selector, mutator, crossover, p_m, p_c, pop_size, elitism,
                                  reproduction, seed, device)
        if path_init_pop and path_rts:
            self.reconstruct = True
            self.path_init_pop = path_init_pop
            self.path_rts = path_rts
            self.history = {}
            self.pop_ids = []
        else:
            self.reconstruct = False

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        # Creates as empty list for the population's representation
        pop_size, pop_repr = self.pop_size, []

        # Recomputes populations' size and extends the list with user-specified initial seed (if any)
        if start_at is not None:
            pop_size -= len(start_at)
            pop_repr.extend(start_at)

        # Initializes pop_size individuals by means of 'initializer' function
        pop_repr.extend(self.initializer(sspace=self.pi.sspace, n_sols=pop_size, device=self.device))

        if self.reconstruct:
            # Stores populations' representation as individual trees (each tree is stored as a .pickle)
            for i, tree in enumerate(pop_repr):
                # Appends representations' ID to the list
                self.pop_ids.append(str(self.seed) + "_" + str(0) + "_" + str(i))  # iteration 0
                # Writes the pickles
                with open(os.path.join(self.path_init_pop, self.pop_ids[-1] + ".pickle"), "wb") as handle:
                    pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Extracts trees' sizes, depths and semantics. From this point on, pop_repr stores trees' semantics only
        pop_size = [len(repr_) for repr_ in pop_repr]
        pop_depth = [_get_tree_depth(repr_) for repr_ in pop_repr]
        pop_repr = [_execute_tree(repr_, self.pi.X) for repr_ in pop_repr]

        # Expands the semantics vectors when individuals are only constants
        pop_repr = [torch.cat(len(self.pi.X)*[repr_[None]]) if len(repr_.shape) == 0 else repr_ for repr_ in pop_repr]

        # Stacks population's sizes, depths and semantics
        pop_size = torch.tensor(pop_size)
        pop_depth = torch.tensor(pop_depth)
        pop_repr = torch.stack(pop_repr)

        # Creates an object of type Population
        self.pop = Population(pop_repr)
        self.pop.size = pop_size
        self.pop.depth = pop_depth

        # Evaluates the population on a given problem instance
        self.pi.evaluate_pop(self.pop)

        # Gets the elite
        self.best_sol = self._get_best_pop(self.pop)

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a GSGP.

        Parameters
        ----------
        n_iter : int (default=20)
            The number of iterations.
        tol : float (default=None)
            Minimum required fitness improvement for n_iter_tol
            consecutive iterations to continue the solve. When best
            solution's fitness is not improving by at least tol for
            n_iter_tol consecutive iterations, the solve will be
            automatically interrupted.
        n_iter_tol : int (default=5)
            Maximum number of iterations to continue the solve while
            not meeting the tol improvement.
        start_at : object (default=None)
            The initial starting point in 𝑆 (it is is assumed to be
            feasible under 𝑆's constraints, if any).
        test_elite : bool (default=False)
            Indicates whether assess the best-so-far solution on the
            test partition (this regards SML-based OPs).
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything.
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness.
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness);
                - log = 3: also, writes elite's representation.
        """
        # Optionally, tracks initialization's timing for console's output
        if verbose > 0:
            start = time.time()

        # 1)
        self._initialize(start_at=start_at)
        # Optionally, evaluates the initial valid solution on the test partition
        if test_elite:
            self.pi.evaluate_sol(self.best_sol, test_elite)

        # Optionally, computes population's AVG and STD (in terms of fitness)
        if log == 2 or verbose == 2:
            self.pop.fit_avg = self.pop.fit.mean().item()
            self.pop.fit_std = self.pop.fit.std().item()

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, self.pop, verbose)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, self.pop, log)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        id_count = 0
        for it in range(1, n_iter + 1, 1):
            id_it = str(self.seed) + "_" + str(it) + "_"
            # 2) 2)
            offs_pop_ids, offs_pop_repr, offs_pop_size, offs_pop_depth, start = [], [], [], [], time.time()

            # 2) 3)
            pop_size = self.pop_size - self.pop_size % 2
            while len(offs_pop_repr) < pop_size:
                # 2) 3) 2)
                p1_idx = p2_idx = self.selector(self.pop, self.pi.min_)
                # Avoids selecting the same parent twice
                while p1_idx == p2_idx:
                    p2_idx = self.selector(self.pop, self.pi.min_)

                # Performs GP-like variation (no reproduction)
                if random.uniform(0, 1) < self.p_c:
                    # 2) 3) 3)
                    offs1_repr, offs2_repr, rt = self.crossover(self.pop[p1_idx], self.pop[p2_idx])
                    offs1_rt_size = offs2_rt_size = len(rt)
                    offs1_rt_depth = offs2_rt_depth = _get_tree_depth(rt)

                    if self.reconstruct:
                        # Stores the random tree as a .pickle
                        rt_id = id_it + "rt_xo_" + str(id_count)
                        id_count += 1
                        with open(os.path.join(self.path_rts, rt_id + ".pickle"), "wb") as handle:
                            pickle.dump(rt, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # Writes the history: when crossover, assumes ms = -1.0
                        offs_pop_ids.append(id_it + "o1_xo_" + str(id_count))
                        id_count += 1
                        self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "crossover",
                                                          "T1": self.pop_ids[p1_idx],
                                                          "T2": self.pop_ids[p2_idx], "Tr": rt_id, "ms": -1.0}
                        offs_pop_ids.append(id_it + "o2_xo_" + str(id_count))
                        id_count += 1
                        self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "crossover",
                                                          "T1": self.pop_ids[p2_idx],
                                                          "T2": self.pop_ids[p1_idx], "Tr": rt_id, "ms": -1.0}
                else:
                    # 2) 3) 4)
                    offs1_repr, rt1, ms1 = self.mutator(self.pop[p1_idx])
                    offs2_repr, rt2, ms2 = self.mutator(self.pop[p2_idx])
                    offs1_rt_size, offs2_rt_size = len(rt1), len(rt2)
                    offs1_rt_depth, offs2_rt_depth = _get_tree_depth(rt1), _get_tree_depth(rt2)

                    if self.reconstruct:
                        # Stores random trees as .pickle and writes the history
                        for rt, offs, p_idx, ms in zip([rt1, rt2], [offs1_repr, offs2_repr], [p1_idx, p2_idx], [ms1, ms2]):
                            rt_id = id_it + "rt_mtn_" + str(id_count)
                            id_count += 1
                            with open(os.path.join(self.path_rts, rt_id + ".pickle"), "wb") as handle:
                                pickle.dump(rt, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            # When mutation, assumes T1 as the parent and T2 as -1.0
                            offs_pop_ids.append(id_it + "o1_mtn_" + str(id_count))
                            id_count += 1
                            self.history[offs_pop_ids[-1]] = {"Iter": it, "Operator": "mutation",
                                                              "T1": self.pop_ids[p_idx], "T2": -1.0,
                                                              "Tr": rt_id, "ms": ms.item()}

                # 2) 3) 5)
                offs_pop_repr.extend([offs1_repr, offs2_repr])
                offs_pop_size.extend([self.pop.size[p1_idx]+offs1_rt_size, self.pop.size[p2_idx]+offs2_rt_size])
                offs_pop_depth.extend([self.pop.depth[p1_idx]+offs1_rt_depth, self.pop.depth[p2_idx]+offs2_rt_depth])

            # Stacks the population's representation, size and depth
            offs_pop_repr = torch.stack(offs_pop_repr)
            offs_pop_size = torch.stack(offs_pop_size)
            offs_pop_depth = torch.stack(offs_pop_depth)

            # 2) 1)
            offs_pop = Population(offs_pop_repr)
            offs_pop.size = offs_pop_size
            offs_pop.depth = offs_pop_depth
            self.pi.evaluate_pop(offs_pop)

            # Updates offspring's history with fitness
            for off, fit in zip(offs_pop_ids, offs_pop.fit):
                self.history[off]["Fitness"] = fit.item()

            # Updates population's IDs
            self.pop_ids = offs_pop_ids

            # Updates the current elite
            best_offs = self._get_best_pop(offs_pop)
            self.best_sol = self._get_best(self.best_sol, best_offs)

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=test_elite)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log == 2 or verbose == 2:
                offs_pop.fit_avg = offs_pop.fit.mean().item()
                offs_pop.fit_std = offs_pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, offs_pop, log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, offs_pop, verbose)

            # Performs population's replacement
            if self.elitism:
                self._elite_replacement(offs_pop, best_offs)
            else:
                self.pop = offs_pop

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

    def write_history(self, path):
        """Writes evolution's history on disk.

        The file that is written will be then used by the
        reconstruction algorithm.

        Parameters
        ----------
        path : str
            File path.
        """
        if self.reconstruct:
            pd.DataFrame.from_dict(self.history, orient="index").to_csv(path)
        else:
            print("Cannot write population's genealogical history since the reconstruction was not activated!")
