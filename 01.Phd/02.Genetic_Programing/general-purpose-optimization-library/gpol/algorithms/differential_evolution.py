import time
import logging

import torch

from gpol.utils.population import Population
from gpol.algorithms.population_based import PopulationBased


class DifferentialEvolution(PopulationBased):
    """Implements Differential Evolution(DE).

    Similarly to Genetic Algorithms (GAs), DE:
        1) is inspired by a biological system;
        2) is a population-based (PB) stochastic iterative solve
         algorithm;
        3) starts with a population of random candidate solutions;
        4) searches for the optimal solution by updating their position
         in the solve space (𝑆) at each iteration/generation.

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased)
        The best solution found.
    pop_size : int (inherited from PopulationBased)
        Population's size.
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    initializer : function (inherited from PopulationBased)
        The initialization procedure.
    selector : function
        The selection procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    crossover : function
        The crossover procedure.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "DifferentialEvolution"

    def __init__(self, pi, initializer, selector, mutator, crossover, m_weights, c_rate=0.8, pop_size=100, seed=0,
                 device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            An instance of OP.
        initializer : function (inherited)
            The initialization procedure.
        selector : function
            The selection procedure.
        mutator : function (inherited from the PopulationBased)
            The mutation procedure.
        crossover : function
            The crossover procedure.
        m_weights : torch.Tensor
            Weights for the differentials in the mutator function. The
            size of the tensor is implicitly related to the number of
            differentials in of the mutation strategy.
        c_rate : float (default=0.8)
            Crossover's rate.
        pop_size : int (default=100)
            The population's size.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)
        self.selector = selector
        self.crossover = crossover
        self.m_weights = m_weights
        self.c_rate = c_rate

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a DE.

        Note that the initial population is generated in a single call
        of the 'initializer' and stored in 'self'.

        This method implements the following pseudo-code:
            1) Create a random initial population of size n (𝑃);
            2) Repeat until satisfying some termination condition,
             typically the number of generations/iterations:
                1) Create an empty population 𝑃’, the population of
                 offsprings;
                2) Repeat until 𝑃’ contains 𝑛 individuals:
                    1) For a given parameter vector from the population
                     of parents (the target), randomly select other
                     three distinct parameter vectors r1, r2 and r3;
                    2) Apply the mutation operator. For example, the
                     traditional mutation operator, often referenced as
                     DE/RAND/1, consists of summing a weighted
                     difference of r2 and r3 to r1 as r1 + F(r2-r3),
                     where F is a mutation factor. The resulting vector
                     is called the donor;
                    3) Apply the crossover operator. For example, the
                     traditional crossover operator consists of
                     developing the trial vector from the elements of
                     the target and donor vectors. The Elements of the
                     donor vector enter the trial vector with a
                     probability cr;
                    4) Evaluate the trial vector;
                    5) Select the most fit candidate-solution from the
                     set [target, trial] and insert it into 𝑃’.
                3) Replace 𝑃 with 𝑃’;
            2) Return the best individual in 𝑃 (the elite).

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
            # 2) 1)
            offs_pop, start = [], time.time()
            # 2) 2)
            for r in range(len(self.pop)):
                target = self.pop.repr_[r].clone()  # solutions' representations are vectors
                # 2) 2) 1)
                parents = self.pop.repr_[self.selector(self.pop)]
                # 2) 2) 2)
                if "target_to_best" in self.mutator.__name__:
                    donor = self.mutator(target, self.best_sol.repr_, parents, self.m_weights)
                elif "best" in self.mutator.__name__:
                    donor = self.mutator(self.best_sol.repr_, parents, self.m_weights)
                else:
                    donor = self.mutator(parents, self.m_weights)
                # 2) 2) 3)
                trial = self.crossover(donor, target, self.c_rate)
                # Temporarily assigns the trial to 𝑃’
                offs_pop.append(trial)

            # If batch training, appends the elite to evaluate_pop it on the same batch(es) as the offspring population
            if self._batch_training:
                offs_pop.append(self.best_sol.repr_)

            # Stacks the population's representation
            offs_pop = Population(torch.stack(offs_pop))

            # 2) 2) 4)
            self.pi.evaluate_pop(offs_pop)

            # Overrides elites's information, if it was re-evaluated, and removes it from 'offsprings'
            if self._batch_training:
                self.best_sol.valid = offs_pop.valid[-1]
                self.best_sol.fit = offs_pop.fit[-1]
                # Removes the elite from the object 'offsprings'
                offs_pop.repr_ = offs_pop.repr_[0: -1]
                offs_pop.valid = offs_pop.valid[0: -1]
                offs_pop.fit = offs_pop.fit[0: -1]

            # 2) 2) 5)
            self._replacement(offs_pop)

            # Finds the best offspring.
            self.best_sol = self._get_best_pop(self.pop)

            # Optionally, evaluates the elite on the test partition
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=test_elite)

            # Optionally, computes iteration's timing
            if (log + verbose) > 0:
                timing = time.time() - start

            # Optionally, computes population's AVG and STD (in terms of fitness)
            if log == 2 or verbose == 2:
                self.pop.fit_avg = self.pop.fit.mean().item()
                self.pop.fit_std = self.pop.fit.std().item()

            # Optionally, writes the log-data on the file
            if log > 0:
                log_event = self._create_log_event(it, timing, self.pop, log)
                logger.info(','.join(list(map(str, log_event))))

            # Optionally, reports the progress on the console
            if verbose > 0:
                self._verbose_reporter(it, timing, self.pop, verbose)

            # Optionally, verifies the tolerance-based stopping criteria
            if tol:
                n_iter_bare, last_fit = self._check_tol(last_fit, tol, n_iter_bare)

                if n_iter_bare == n_iter_tol:
                    break

    def _replacement(self, offs_pop):
        """Performs DE-like replacement of 𝑃 with 𝑃’.

        Implements an efficient (vectorized) selection of the most fit
        candidate-solutions between 𝑃 and 𝑃’.

        Parameters
        ----------
        offs_pop : torch.Tensor
            The population of offsprings at iteration i.
        """
        # Concatenates tensors into a 2 column matrix
        t = torch.cat((self.pop.fit[:, None], offs_pop.fit[:, None]), dim=1)
        # Selects the indexes of the most fit solutions
        idxs = torch.min(t, dim=1)[1] if self.pi.min_ else torch.max(t, dim=1)[1]
        idxs_targets, idxs_trials = (1 - idxs).nonzero(), idxs.nonzero()
        # Gets representations of the most fit solutions
        pop = Population(torch.squeeze(torch.cat([self.pop.repr_[idxs_targets].clone(),
                                                  offs_pop.repr_[idxs_trials].clone()])))
        # Gets fitness-cases of the most fit solutions
        pop.fit = torch.squeeze(torch.cat([self.pop.fit[idxs_targets].clone(), offs_pop.fit[idxs_trials].clone()]))
        # Gets validity states of the most fit solutions.
        if hasattr(self.pop, "valid"):
            pop.valid = torch.squeeze(torch.cat([self.pop.valid[idxs_targets].clone(),
                                                 offs_pop.valid[idxs_trials].clone()]))
        # Replaces 𝑃 with 𝑃’
        self.pop = pop
