import time
import random
import logging

import torch

from gpol.utils.population import Population
from gpol.utils.solution import Solution
from gpol.algorithms.random_search import RandomSearch

from gpol.problems.knapsack import Knapsack01


class HillClimbing(RandomSearch):
    """Hill Climbing (HC) Algorithm.

    The Local Search (LS) can be seen among the first attempts to
    improve upon the RS by introducing some intelligence in the
    solve process. It relies upon the concept of the neighborhood
    and, at each iteration, a limited number of neighbors of the best
    best-so-far solution is explored. Usually, the LS algorithms divide
    in two branches: Hill Climbing (HC) and Simulated Annealing (SA).
    This class implements HC algorithm Climbing (Hill Descent for
    minimization problems), where the best-so-far solution is replaced
    by its neighbor when the latter is at least as good as the former.
    An instance of a HC can be characterized by the following features:
        1) an IP (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point in 𝑆;
        3) the best solution found by the solve procedure;
        4) a procedure to generate neighbours of a given solution (the
         neighbour-generation function);
        5) the number of neighbours to be explored at each iteration
         (the neighbourhood size);
        6) a random state for random numbers generation;
        7) the processing device (CPU or GPU).

        To solve a PI, the LS:
        1) initializes the solve at a given point in 𝑆 (normally, by
         sampling candidate solution(s) at random);
        2) searches throughout 𝑆, in iterative manner, for the best
         possible solution by sampling a set of neighbors of the
         current best solution at a given iteration, using a
         neighborhood function, and choosing the one with the best
         fitness. Traditionally, the termination condition for an ISA
         is the number of iterations, the default stopping criteria in
         this library.

    Attributes
    ----------
    pi : Problem (inherited from RandomSearch)
        An instance of OP.
    initializer : function (inherited from RandomSearch)
        The initialization procedure.
    best_sol : Solution (inherited from RandomSearch)
        The best solution found.
    nh_function : function
        The neighbour-generation procedure.
    nh_size : int
        The neighborhood size of a given solution at a given iteration.
    seed : int (inherited from RandomSearch)
        The seed for random numbers generators.
    device : str (inherited from RandomSearch)
        Specification of the processing device.
    """
    __name__ = "HillClimbing"

    def __init__(self, pi, initializer, nh_function, nh_size=100, seed=0, device="cpu"):
        RandomSearch.__init__(self, pi, initializer, seed, device)
        """Objects' constructor.        

        Parameters
        ----------
        pi : Problem
            An instance of OP.
        initializer : function
            The initialization procedure.
        nh_function : function
            The neighbour-generation procedure.
        nh_size : int (default=100)
            The neighborhood size of a given solution at a given iteration.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        self.nh_function = nh_function
        self.nh_size = nh_size

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Implements the solve procedure of a LS algorithm.

        This method follows the pseudo-code of a LS algorithm, provided
        in below:
            1) Initialize: generates one random (valid) initial
             solution 𝑖 (the initializer is called in main method);
            2) Repeat until satisfying some stopping criterion (for
             example, the number of iterations):
                2) 1) Generate nh_size neighbors of 𝑖;
                2) 2) Select the best solution 𝑗 from 2) 1);
                2) 3) If the fitness of solution 𝑗 is better or equal
                 than the fitness of solution 𝑖, then 𝑖=𝑗.

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
                - verbose = 0: do not print anything;
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness;
                - verbose = 2: also prints neighborhood's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes neighborhood's average and
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

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, None, 1)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, None, log)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        for it in range(1, n_iter + 1):
            # 2) 1)
            neighborhood, start = [], time.time()
            # Generates nh_size neighbors of self.best_sol
            for _ in range(self.nh_size):
                neighborhood.append(self.nh_function(self.best_sol.repr_))
            # If batch training, appends the best-so-far to evaluate_pop it on the same batch(es) as its neighbors
            if self._batch_training:
                neighborhood.append(self.best_sol.repr_)
            # If solutions are objects of type torch.Tensor, stacks their representations in the same tensor
            if isinstance(self.best_sol.repr_, torch.Tensor):
                neighborhood = torch.stack(neighborhood)
            # Creates an object of type Population for the sake of efficient evaluation of the whole set of solutions
            neighborhood = Population(neighborhood)
            # Evaluates the neighborhood
            self.pi.evaluate_pop(neighborhood)
            # If batch training, overrides the fitness of the best-so-far and removes it from the neighborhood
            if self._batch_training:
                self.best_sol.fit = neighborhood.fit[-1]
                neighborhood.repr_ = neighborhood.repr_[0: -1]
                neighborhood.valid = neighborhood.valid[0: -1]
                neighborhood.fit = neighborhood.fit[0: -1]

            # 2) 2)
            best_neighbor = self._get_best_nh(neighborhood)

            # 2) 3)
            self.best_sol = self._get_best(self.best_sol, best_neighbor)

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=test_elite)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes neighborhood's AVG and STD (in terms of fitness)
            if log == 2 or verbose == 2:
                neighborhood.fit_avg = neighborhood.fit.mean().item()
                neighborhood.fit_std = neighborhood.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, neighborhood, log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, neighborhood, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

    def _get_best_nh(self, neighborhood):
        """Returns the best neighbor from the neighborhood.

        Parameters
        ----------
        neighborhood : Population
            An object of type Population which represents the set of
            neighbors of the best-so-far solution, after the evaluation.

        Returns
        -------
        Solution
            The best neighbor from the neighborhood.
        """
        # Finds the index of the best neighbor in the neighborhood
        if self.pi.min_:
            idx_best_nh = neighborhood.fit.argmin()
        else:
            idx_best_nh = neighborhood.fit.argmax()

        # Copies solution's first representation
        if isinstance(neighborhood.repr_, torch.Tensor):
            best_nh = Solution(neighborhood.repr_[idx_best_nh].clone())  # if it is a tensor
        else:
            best_nh = Solution(neighborhood.repr_[idx_best_nh].copy())  # if it is a list

        # Copies neighbor's fitness value(s) and validity state
        best_nh.fit = neighborhood.fit[idx_best_nh].clone()
        if hasattr(neighborhood, "test_fit"):
            best_nh.test_fit = neighborhood.test_fit[idx_best_nh].clone()
        if hasattr(neighborhood, "valid"):
            best_nh.valid = neighborhood.valid[idx_best_nh]

        return best_nh

    def _create_log_event(self, it, timing, neighborhood, log):
        """Implements a standardized log-event.

        Creates a log-event for the underlying best-so-far solution.

        Parameters
        ----------
        it : int
            Iteration's number.
        timing : float
            Iterations's running time in seconds.
        neighborhood : Population
            An object of type Population which represents the current
            neighborhood.
        log : int, optional
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes population's average and
                    standard deviation (in terms of fitness);
                - log = 3: also, writes elite's representation.
        """
        # Appends the current iteration, its timing, and elite's length and fitness
        log_event = [it, timing, len(self.best_sol), self.best_sol.fit.item()]
        if hasattr(self.best_sol, 'test_fit'):
            log_event.append(self.best_sol.test_fit.item())
        # Also, writes population's average and standard deviation (in terms of fitness)
        if log >= 2:
            if neighborhood:
                log_event.extend([neighborhood.fit_avg, neighborhood.fit_std])
            else:
                log_event.extend([-1.0, -1.0])  # special case for for iteration 0
        # Also, writes elite's representation
        if log >= 3:
            log_event.append(self.best_sol.repr_)

        return log_event

    def _verbose_reporter(self, it, timing, neighborhood, verbose=0):
        """Reports the progress of the solve on the console.

        Parameters
        ----------
        it : int
            Integer that represents the current iteration.
        timing : float
            Floating-point that represents the processing time of the
            current iteration.
        neighborhood : Population
            An object of type Population that represents the current
            neighborhood.
        verbose : int, optional (default=0)
            An integer that controls the verbosity of the solve. The
            following nomenclature is applied in this class:
                - verbose = 0: do not print anything (controlled from
                 the solve method);
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness (default);
                - verbose = 2: also prints neighborhood's average
                    and standard deviation (in terms of fitness).
        """
        if it == -1:
            if hasattr(self.best_sol, "test_fit"):
                print('-' * 103)
                print(' ' * 11 + '|{:^53}  |{:^34}|'.format("Best solution", "Neighborhood"))
                print('-' * 103)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', 'Test Fitness',
                                         "Timing", "|", "AVG Fitness", "STD Fitness"))
                print('-' * 103)
            else:
                print('-' * 86)
                print(' ' * 11 + '|{:^36}  |{:^34}|'.format("Best solution", "Neighborhood"))
                print('-' * 86)
                line_format = '{:<10} {:<1} {:<8} {:<16} {:>10} {:<1} {:<16} {:>16}'
                print(line_format.format('Generation', "|", 'Length', 'Fitness', "Timing",
                                         "|", "AVG Fitness", "STD Fitness"))
        else:
            if hasattr(self.best_sol, "test_fit"):
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                length = len(self.best_sol)
                if verbose == 2:
                    fit_avg, fit_std = neighborhood.fit_avg, neighborhood.fit_std
                else:
                    fit_avg, fit_std = -1.0, -1.0
                print(line_format.format(it, "|", length, self.best_sol.fit, self.best_sol.test_fit, timing, "|",
                                         fit_avg, fit_std))
            else:
                line_format = '{:<10d} {:<1} {:<8d} {:<16g} {:>10.3f} {:<1} {:<16g} {:>16g}'
                # If the the type of OP is of knapsack's family, then sum the vector, otherwise the length
                length = int(self.best_sol.repr_.sum().item()) if isinstance(self.pi, Knapsack01) else len(self.best_sol)
                if verbose == 2:
                    fit_avg, fit_std = neighborhood.fit_avg, neighborhood.fit_std
                else:
                    fit_avg, fit_std = -1.0, -1.0
                print(line_format.format(it, "|", length, self.best_sol.fit, timing, "|", fit_avg, fit_std))


class SimulatedAnnealing(HillClimbing):
    """Simulated Annealing (SA) Algorithm.

    HC suffers from several limitations, namely it frequently gets
    stuck at a locally optimal point. To overcome this issue, the
    scientific community have proposed several approaches, among
    them the SA algorithm.

    As the HC, the SA relies on the concept of neighborhood. One thing
    that distinguishes SA from HC is an explicitly defined ability to
    escape from a local optima. This ability was conceived by
    simulating, in the computer, a well known phenomenon in metallurgy
    called annealing (reason why the algorithm is called Simulated
    Annealing). Following what happens in metallurgy, in SA, the
    transition from the current state (the best-so-far candidate
    solution 𝑖), to a candidate new state (𝑗, a neighbor of 𝑖), happens
    only for two reasons: either because the candidate state is better
    or following the outcome of an acceptance probability function - a
    function which probabilistically accepts a transition towards the
    candidate new state, even if it is worse than the current state,
    depending on the states' energy (fitness) and a global
    (time-decreasing) parameter called the temperature (t). The
    acceptance function has the following characteristics: the larger
    the energy differential (|f(𝑖) - f(𝑗)|, the smaller the
    acceptance probability; the larger the temperature parameter (t),
    the larger the acceptance probability; the larger the decrease rate
    of the temperature of the time (iterations), the smaller the
    acceptance probability over the time.

    Under this perspective, SA can be seen as an attempt to improve
    upon HC by adding more intelligence in the solve strategy. The
    code contained in this class implements the SA algorithm. An
    instance of a SA can be characterized by the following features:
        1) an instance of an OP (i.e., what to solve/optimize);
        2) a function to initialize the solve at a given point in 𝑆;
        3) the best solution found by the solve procedure;
        4) a procedure to generate neighbours of a given solution (the
         neighbour-generation function);
        5) the number of neighbours to be explored at each iteration
         (the neighbourhood size);
        6) the control temperature;
        7) the update rate of 6);
        8) a random state for random numbers generation;
        9) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from HillClimbing)
        An instance of OP.
    initializer : function (inherited from HillClimbing)
        The initialization procedure.
    best_sol : Solution (inherited from HillClimbing)
        The best solution found.
    nh_function : function (inherited from HillClimbing)
        The neighbour-generation procedure.
    nh_size : int (inherited from HillClimbing)
        The neighborhood size of a given solution at a given iteration.
    control : float
        The control temperature.
    update_rate : float
        The update rate of the control temperature.
    seed : int (inherited from HillClimbing)
        The seed for random numbers generators.
    device : str (inherited from HillClimbing)
        Specification of the processing device.
    """
    __name__ = "SimulatedAnnealing"

    def __init__(self, pi, initializer, nh_function, nh_size=100, control=1.0, update_rate=0.9, seed=0, device="cpu"):
        HillClimbing.__init__(self, pi, initializer, nh_function, nh_size, seed, device)
        """Objects' constructor. 
 
        Parameters
        ----------
        pi : Problem
            An instance of OP.
        initializer : function
            The initialization procedure.
        nh_function : function
            The neighbour-generation procedure.
        nh_size : int (default=100)
            The neighborhood size of a given solution at a given 
            iteration.
        control : float (default=1.0)
            The control temperature.
        update_rate : float (default=0.9)
            The update rate of the control temperature.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.        
        """
        self.control = control
        self.update_rate = update_rate

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a SA.

        This method follows the pseudo-code of a LS algorithm, provided
        in below:
            1) Initialize: generate one random (valid) initial solution 𝑖;
            2) Repeat: until satisfying some stopping criterion (for example, number of iterations):
                2) 1) repeat 𝐿_𝑘 times (where 𝐿_𝑘 stands for the number of neighbors, a.k.a. transitions):
                    2) 1) 1) choose a random neighbor 𝑗 ∈ 𝑁(𝑖),
                    2) 1) 2) if f(j) ≥ f(i) then i = j;
                    2) 1) 3) else if  e^(−(f(i)−f(j))/c) > r then i=j, r ∈ [0,1[, f(i) > f(j) and f(j) is valid.
                2) 2) update 𝑐 (the control parameter a.k.a. temperature).

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
                - verbose = 0: do not print anything;
                - verbose = 1: prints current iteration, its timing,
                    and elite's length and fitness;
                - verbose = 2: also prints neighborhood's average
                    and standard deviation (in terms of fitness).
        log : int, optional (default=0)
            An integer that controls the verbosity of the log file. The
            following nomenclature is applied in this class:
                - log = 0: do not write any log data;
                - log = 1: writes the current iteration, its timing, and
                    elite's length and fitness;
                - log = 2: also, writes neighborhood's average and
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

        # Optionally, reports initializations' summary results on the console
        if verbose > 0:
            # Creates reporter's header reports the result of initialization
            self._verbose_reporter(-1, 0, None, 1)
            self._verbose_reporter(0, time.time() - start, None, 1)

        # Optionally, writes the log-data
        if log > 0:
            log_event = [self.pi.__name__, self.__name__, self.seed]
            logger = logging.getLogger(','.join(list(map(str, log_event))))
            log_event = self._create_log_event(0, 0, None, log)
            logger.info(','.join(list(map(str, log_event))))

        # Optionally, creates local variables to account for the tolerance-based stopping criteria
        if tol:
            n_iter_bare, last_fit = 0, self.best_sol.fit.clone()

        # 2)
        best_sol_ = self.best_sol
        for it in range(1, n_iter + 1):
            # 2) 1) from HC
            neighborhood, start = [], time.time()
            # Generates nh_size neighbors
            for _ in range(self.nh_size):
                neighborhood.append(self.nh_function(self.best_sol.repr_))
            # If batch training, appends the best-so-far to evaluate_pop it on the same batch(es) as its neighbors
            if self._batch_training:
                neighborhood.append(self.best_sol.repr_)
            # If solutions are objects of type torch.Tensor, stacks the representations in the same tensor
            if isinstance(self.best_sol.repr_, torch.Tensor):
                neighborhood = torch.stack(neighborhood)
            # Creates an object of type Population, for the sake of efficient evaluation of the whole set of solutions
            neighborhood = Population(neighborhood)
            # Evaluates the neighborhood
            self.pi.evaluate_pop(neighborhood)
            # If batch training, overrides the fitness of the best-so-far and removes it from the neighborhood object
            if self._batch_training:
                self.best_sol.fit = neighborhood.fit[-1]
                neighborhood.repr_ = neighborhood.repr_[0: -1]
                neighborhood.valid = neighborhood.valid[0: -1]
                neighborhood.fit = neighborhood.fit[0: -1]

            # 2) 2) from HC
            best_neighbor = self._get_best_nh(neighborhood)

            # 2) 3) from HC
            self.best_sol = self._get_best(self.best_sol, best_neighbor)

            # If no neighbor 𝑗 is at least as good as 𝑖, run acceptance probability function in 𝑁(𝑖)
            if id(self.best_sol) != id(best_neighbor):
                # 2) 1) 1)
                for nh_i in range(len(neighborhood)):
                    # 2) 1) 3)
                    inversion_c = - 1.0 if self.pi.min_ else 1
                    p_accept = torch.exp((-inversion_c * (self.best_sol.fit - neighborhood.fit[nh_i])) / self.control)
                    if p_accept > random.random():
                        self.best_sol = Solution(neighborhood[nh_i])
                        # Copies neighbor's fitness value(s) and validity state
                        self.best_sol.fit = neighborhood.fit[nh_i].clone()
                        if hasattr(neighborhood, "test_fit"):
                            self.best_sol.test_fit = neighborhood.test_fit[nh_i].clone()
                        if hasattr(neighborhood, "valid"):
                            self.best_sol.valid = neighborhood.valid[nh_i]
                        break

            # 2) 2) from SA
            self.control *= self.update_rate

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=test_elite)

            # Updates global best solution
            best_sol_ = self._get_best(self.best_sol, best_sol_)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes neighborhood's AVG and STD (in terms of fitness)
            if log == 2 or verbose == 2:
                neighborhood.fit_avg = neighborhood.fit.mean().item()
                neighborhood.fit_std = neighborhood.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, neighborhood, log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, neighborhood, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

        self.best_sol = best_sol_
