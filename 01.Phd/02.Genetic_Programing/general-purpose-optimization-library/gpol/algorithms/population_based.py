import torch

from gpol.problems.knapsack import Knapsack01

from gpol.utils.solution import Solution
from gpol.utils.population import Population
from gpol.algorithms.random_search import RandomSearch


class PopulationBased(RandomSearch):
    """Population-based ISA (PB-ISAs).

    Based on the number of candidate solutions they handle at each
    step, the optimization algorithms can be categorized into
    Single-Point (SP) and Population-Based (PB) approaches. The solve
    procedure in the SP algorithms is generally guided by the
    information provided by a single candidate solution from 𝑆,
    usually the best-so-far solution, that is gradually evolved in a
    well defined manner in hope to find the global optimum. The HC is
    an example of a SP algorithm as the solve is performed by
    exploring the neighborhood of the current best solution.
    Contrarily, the solve procedure in PB algorithms is generally
    guided by the information shared by a set of candidate solutions
    and the exploitation of its collective behavior of different ways.
    In abstract terms, one can say that PB algorithms share, at least,
    the following two features: an object representing the set of
    simultaneously exploited candidate solutions (i.e., the
    population), and a procedure to "move" them across 𝑆.

    An instance of a PB-ISA is characterized by the following features:
        1) a PI (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point of the
         solve space (𝑆);
        3) the best solution found by the PB-ISA;
        4) the number of simultaneously exploited solution (the
         population's size);
        6) a collection of candidate solutions - the population;
        7) a random state for random numbers generation;
        8) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from RandomSearch)
        An instance of OP.
    pop_size : int
        The population's size.
    best_sol : Solution (inherited from RandomSearch)
        The best solution found.
    pop_size : int
        Population's size.
    pop : Population
        Object of type Population which holds population's collective
        representation, feasibility states and fitness values.
    initializer : function (inherited from RandomSearch)
        The initialization procedure.
    mutator : function
        A function to move solutions across 𝑆.
    seed : int (inherited from RandomSearch)
        The seed for random numbers generators.
    device : str (inherited from RandomSearch)
        Specification of the processing device.
    """

    def __init__(self, pi, initializer, mutator, pop_size=100, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            Instance of an optimization problem (PI).
        initializer : function
            The initialization procedure.
        mutator : function
            A function to move solutions across the solve space.
        pop_size : int (default=100)
            Population's size.
        seed : int str (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        RandomSearch.__init__(self, pi, initializer, seed, device)
        self.mutator = mutator
        self.pop_size = pop_size
        # Initializes the population's object (None by default)
        self.pop = None

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

        # Recomputes populations' size and extends the list with user-specified initial seed, is such exists
        if start_at is not None:
            pop_size -= len(start_at)
            pop_repr.extend(start_at)

        # Initializes pop_size individuals by means of 'initializer' function
        pop_repr.extend(self.initializer(sspace=self.pi.sspace, n_sols=pop_size, device=self.device))

        # Stacks population's representation, if candidate solutions are objects of type torch.Tensor
        if isinstance(pop_repr[0], torch.Tensor):
            pop_repr = torch.stack(pop_repr)

        # Creates an object of type 'Population', given the initial representation
        self.pop = Population(pop_repr)

        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)

        # Gets the best in the initial population
        self.best_sol = self._get_best_pop(self.pop)

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a PB-ISA.

        This method implements the pseudo-code of a given PB-ISA.

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
        pass

    def _get_best_pop(self, pop):
        """Returns the best solution in a given population.

        Parameters
        ----------
        pop : Population
            An object of type Population (already evaluated).

        Returns
        -------
        Solution
            The best candidate solution in a given population.
        """
        # Finds the index of the best candidate-solution
        if self.pi.min_:
            idx = pop.fit.argmin()
        else:
            idx = pop.fit.argmax()

        # Copies solution's representation
        if isinstance(pop.repr_, torch.Tensor):
            best_sol = Solution(pop.repr_[idx].clone())
        else:
            best_sol = Solution(pop.repr_[idx].copy())

        # Copies solution's validity state and fitness values
        best_sol.fit = pop.fit[idx].clone()
        if hasattr(pop, "test_fit"):
            best_sol.test_fit = pop.test_fit[idx].clone()
        if hasattr(pop, "valid"):
            best_sol.valid = pop.valid[idx]
        if hasattr(pop, "size"):
            best_sol.size = pop.size[idx].clone()
        if hasattr(pop, "depth"):
            best_sol.depth = pop.depth[idx].clone()

        return best_sol

    def _get_worst_pop(self, pop):
        """Returns the worst solution in a given population.

        Parameters
        ----------
        pop : Population
            An object of type Population (already evaluated).

        Returns
        -------
        Solution
            The worst candidate-solution in a given population.
        """
        # Finds the index of the worst candidate-solution
        if self.pi.min_:
            idx = pop.fit.argmax()
        else:
            idx = pop.fit.argmin()

        # Copies solution's representation
        if isinstance(pop.repr_, torch.Tensor):
            worst_sol = Solution(pop.repr_[idx].clone())
        else:
            worst_sol = Solution(pop.repr_[idx].copy())

        # Copies solution's validity state and fitness values
        worst_sol.fit = pop.fit[idx].clone()
        if hasattr(pop, "test_fit"):
            worst_sol.test_fit = pop.test_fit[idx].clone()
        if hasattr(pop, "valid"):
            worst_sol.valid = pop.valid[idx]

        return worst_sol

    def _create_log_event(self, it, timing, pop, log):
        """Implements a standardized log-event

        Creates a log-event for the underlying best-so-far solution.

        Parameters
        ----------
        it : int
            Iteration's number.
        timing : float
            Iterations's running time in seconds.
        pop : Population
            An object of type Population which represents the current
            population (at the end of iteration i).
        log : int, optional
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data.
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness.
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness).
                - log = 3: also, writes elite's representation
        """
        # Appends the current iteration, its timing, and elite's length and fitness
        log_event = [it, timing, len(self.best_sol), self.best_sol.fit.item()]
        if hasattr(self.best_sol, 'test_fit'):
            log_event.append(self.best_sol.test_fit.item())
        if log >= 2:
            log_event.extend([pop.fit_avg, pop.fit_std])
        # Also, writes elite's representation
        if log >= 3:
            log_event.append(self.best_sol.repr_)

        return log_event

    def _verbose_reporter(self, it, timing, pop, verbose=0):
        """Reports the progress of the solve on the console.

        Parameters
        ----------
        it : int
            Integer that represents the current iteration.
        timing : float
            Floating-point that represents the processing time of the
            current iteration.
        pop : Population
            An object of type Population that represents the current
            population/swarm.
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything (controlled from
                 the solve method).
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness (default).
                - verbose = 2: also prints population's average
                    and standard deviation (in terms of fitness).
        """
        if it == -1:
            if hasattr(self.best_sol, "test_fit"):
                print('-' * 103)
                print(' ' * 11 + '|{:^53}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 103)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', 'Test Fitness',
                                         "Timing", "|", "AVG Fitness", "STD Fitness"))
                print('-' * 103)
            else:
                print('-' * 86)
                print(' ' * 11 + '|{:^36}  |{:^34}|'.format("Best solution", "Population"))
                print('-' * 86)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', "Timing",
                                         "|", "AVG Fitness", "STD Fitness"))
        else:
            if hasattr(self.best_sol, "test_fit"):
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                length = len(self.best_sol)
                if verbose == 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0
                print(line_format.format(it, "|", length, self.best_sol.fit, self.best_sol.test_fit, timing, "|",
                                         avgfit, stdfit))
            else:
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                # If the the type of OP is of knapsack's family, then sum the vector, otherwise the length
                length = int(self.best_sol.repr_.sum().item()) if isinstance(self.pi, Knapsack01) else len(self.best_sol)
                if verbose == 2:
                    avgfit, stdfit = pop.fit_avg, pop.fit_std
                else:
                    avgfit, stdfit = -1.0, -1.0
                print(line_format.format(it, "|", length, self.best_sol.fit, timing, "|", avgfit, stdfit))

    @staticmethod
    def _get_phen_div(pop):
        """Returns the phenotypic diversity of a population.

        Parameters
        ----------
        pop : Population
            An object of type population, after evaluation.

        Returns
        -------
        torch.Tensor
            The standard deviation of population's fitness values.
        """
        return torch.std(pop.fit)
