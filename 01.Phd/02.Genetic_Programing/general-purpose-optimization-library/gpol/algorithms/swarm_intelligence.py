import time
import logging

import torch

from gpol.utils.population import Population
from gpol.utils.solution import Solution
from gpol.algorithms.population_based import PopulationBased


class SPSO(PopulationBased):
    """Synchronous Particle Swarm Optimization (S-PSO).

    Similarly to Genetic Algorithms (GAs), the PSO:
        1) is inspired by a biological system;
        2) is a population-based (PB) stochastic iterative solve
         algorithm;
        3) starts with a population of random candidate solutions;
        4) searches for the optimal solution by updating their position
         in the solve space (𝑆) at each iteration/generation.

    Differently from GAs, the position's update of candidate solutions
    is performed by means of one single operator - the force generating
    mechanism (a.k.a. update rule).

    "In the original PSO algorithm, the particles’ velocities and
    positions are updated after the whole swarm performance is
    evaluated. This variant of the algorithm is also known as
    Synchronous PSO (S-PSO). The strength of this update method is in
    the exploitation of the information (...)
    Because the 𝑝Best_𝑖 and 𝑔Best are updated after all the particles
    are evaluated, S-PSO ensures that all the particles receive perfect
    and complete information about their neighbourhood, leading to a
    better choice of 𝑔Best and thus allowing the particles to exploit
    this information so that a better solution can be found. However,
    this possibly leads the particles in S-PSO to converge faster,
    resulting in a premature convergence."
        - A Synchronous-Asynchronous PSO, Ab Aziz et al. (2014)

    The code contained in this class implements S-PSO that can be
    characterized by the following features:
        1) an instance of an OP, i.e., what to solve/optimize;
        2) a function to initialize the solve at a given point in 𝑆;
        3) a function to mutate candidate-solutions, i.e., to move them
         across 𝑆. In the scope of this library, this regards the force
         generating mechanism which encapsulates the logic of S-PSO.
        4) velocity's bounds (aka clamp);
        4) the population's size;
        5) the best solution found by the PB-ISA;
        6) a collection of candidate solutions - the population;
        7) a random state for random numbers generation;
        8) the processing device (CPU or GPU).

    Attributes
    ----------
    pi : Problem (inherited from PopulationBased)
        An instance of OP.
    best_sol : Solution (inherited from PopulationBased)
        The best solution found.
    initializer : function (inherited from PopulationBased)
        The initialization procedure.
    mutator : function (inherited from PopulationBased)
        The mutation procedure.
    v_clamp : float
        Velocity's clamping bound. If not None, 𝒗_𝒑(𝒕)  is not
        clamped; otherwise will be clamped at +/- v_clamp.
    pop_size : int (inherited from PopulationBased)
        The population's size (a.k.a., swarm's size).
    pop : Population (inherited from PopulationBased)
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    seed : int (inherited from PopulationBased)
        The seed for random numbers generators.
    device : str (inherited from PopulationBased)
        Specification of the processing device.
    """
    __name__ = "S-PSO"

    def __init__(self, pi, initializer, mutator, v_clamp=4.0, pop_size=100, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            Optimization problem's instance (PI).
        initializer : function
            The initialization procedure.
        mutator : function
            A function to move solutions across the solve space.
        v_clamp : float (default=4.0)
            Velocity's clamping bound. If not None, 𝒗_𝒑(𝒕)  is not
            clamped; otherwise will be clamped at +/- v_clamp.
        pop_size : int (default=100)
            Population's size.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        PopulationBased.__init__(self, pi, initializer, mutator, pop_size, seed, device)
        self.v_clamp = v_clamp

    def _initialize(self, start_at=None):
        """Initializes the solve at a given point in 𝑆.

        Note that the user-specified start_at is assumed to be feasible
        under 𝑆's constraints.

        Parameters
        ----------
        start_at : object (default=None)
            A user-specified initial starting point in 𝑆.
        """
        if start_at is not None:
            # Recomputes populations' size and generates initial solutions' representations
            self.pop = Population(self.initializer(self.pi.sspace, self.pop_size - len(start_at), self.device))
            # Extends the representation with the user-specified initial seed
            self.pop.repr_ = torch.cat([self.pop.repr_, start_at])
        else:
            self.pop = Population(self.initializer(self.pi.sspace, self.pop_size, self.device))

        # Evaluates population on the problem instance
        self.pi.evaluate_pop(self.pop)
        # Defines the PSO-specific features of a population (swarm)
        self.pop.velocity = torch.zeros((self.pop_size, self.pi.sspace["n_dims"]), device=self.device)
        self.pop.lBest_repr_ = self.pop.repr_.clone()
        self.pop.lBest_fit = self.pop.fit.clone()
        self.best_sol = self._get_best_pop(self.pop)

    def solve(self, n_iter=20, tol=None, n_iter_tol=5, start_at=None, test_elite=False, verbose=0, log=0):
        """Defines the solve procedure of a S-PSO.

        This method implements the following pseudo-code:
            1) Create a random initial swarm of size n (Sw);
            2) Repeat until satisfying some termination condition,
             typically the number of iterations:
                1) Calculate fitness ∀ particle p in Sw;
                2) Update local best ∀ p in Sw (cognitive factor);
                3) Update global best in Sw (social factor);
                4) Update velocity ∀ p in Sw *;
                5) Update position ∀ p in Sw **;
            3) Return the best individual in 𝑃 (the elite).

        * 𝒗_𝒑(𝒕)=𝒘∗𝒗_𝒑(𝒕−𝟏)+𝑪_𝟏∗𝑹_𝟏∗𝒈𝑩𝒆𝒔𝒕−𝒑𝒐𝒔_𝒑(𝒕−𝟏)+𝑪_𝟐∗𝑹_𝟐∗𝒍𝑩𝒆𝒔𝒕_𝒑−𝒑𝒐𝒔_𝒑(𝒕−𝟏)
        ** 𝒑𝒐𝒔_𝒑(𝒕)=𝒑𝒐𝒔_𝒑(𝒕−𝟏)+𝒗_𝒑(𝒕)

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
            start = time.time()
            self._update(it, n_iter, test_elite)

            # Evaluates the elite on the test partition, if necessary
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

    def _update(self, it=0, it_max=20, test_elite=False):
        """Executes step 2) of PSO algorithm.

        For the sake of computational efficiency (to allow vectorized
        operations), the whole set of sub-steps of 2) was placed in
        this method. First, PSO's force generating mechanism (aka
        update rule) is applied, for the whole swarm at once; then
        particles' positions are collectively updated; after that,
        particles are evaluated and the respective local best are
        updated. Notice that the update of the global best is performed
        after all the swarm gets updated and evaluated (S-PSO).

        Parameters
        ----------
        it : int (default=0)
            Current iteration. Required to adjust the inertia weight.
        it_max : int (default=20)
            Maximum amount of iterations. Required to adjust the
            inertia weight.
        test_elite : bool (default=False)
            A flag indicating whether the best-so-far solution (𝑖)
            should be evaluated on the test partition (regards SML-OPs).
        """
        # 2) 4)
        velocities = self.mutator(self.pop.repr_, self.pop.velocity, self.pop.lBest_repr_, self.best_sol.repr_, it,
                                  it_max)
        # Optionally, clamps the new velocity vector
        if self.v_clamp is not None or self.v_clamp > 0.0:
            velocities[torch.where(-self.v_clamp < velocities, 0, 1).bool()] = -self.v_clamp
            velocities[torch.where(velocities < self.v_clamp, 0, 1).bool()] = self.v_clamp

        # Overrides velocity
        self.pop.velocity = velocities
        # 2) 5)
        self.pop.repr_ += velocities

        # If batch training, appends the elite to evaluate_pop it on the same batch(es) as the offspring population
        if self._batch_training:
            self.pop.repr_ = torch.cat([self.pop.repr_, self.best_sol.repr_[None, :]])

        # 2) 1)
        self.pi.evaluate_pop(self.pop)

        # Overrides elites's information, if it was re-evaluated, and removes it from 'offsprings'
        if self._batch_training:
            # Updates elite's fitness
            self.best_sol.fit = self.pop.fit[-1]
            # Removes the elite from the offsprings
            self.pop.repr_ = self.pop.repr_[0:-1]
            self.pop.valid = self.pop.valid[0: -1]
            self.pop.fit = self.pop.fit[0:-1]

        # 2) 2)
        if self.pi.min_:
            where_ = torch.where(self.pop.fit > self.pop.lBest_fit, 1, 0)
        else:
            where_ = torch.where(self.pop.fit < self.pop.lBest_fit, 1, 0)

        # 2) 2) Update LBests' fitness cases
        stack_ = torch.stack([self.pop.fit, self.pop.lBest_fit], dim=1)
        self.pop.lBest_fit = stack_[torch.arange(0, self.pop_size, device=self.device), where_]
        # 2) 2) Update LBests' representations
        stack_ = torch.stack([self.pop.repr_, self.pop.lBest_repr_], dim=1)
        self.pop.lBest_repr_ = stack_[torch.arange(0, self.pop_size, device=self.device), where_].clone()

        # 2) 3)
        self.best_sol = self._get_best(self.best_sol, self._get_best_pop(self.pop))

        # Optionally, evaluates the elite on the test set
        if test_elite:
            self.pi.evaluate_sol(self.best_sol, test=test_elite)


class APSO(SPSO):
    """Asynchronous Particle Swarm Optimization (A-PSO)

    Carlisle and Dozier have proposed an Asynchronous update (A-PSO),
    where global best is identified immediately after updating the
    position of each particle. Hence, particles are updated using
    incomplete information, enhancing algorithm's exploratory features.

    Attributes
    ----------
    pi : Problem (inherited)
        An instance of OP.
    best_sol : Solution (inherited)
        The best solution found.
    initializer : function (inherited)
        The initialization procedure.
    mutator : function (inherited)
        The mutation procedure.
    v_clamp : float
        Velocity's clamping bound. If not None, 𝒗_𝒑(𝒕)  is not
        clamped; otherwise will be clamped at +/- v_clamp.
    pop_size : int
        The population's size.
    pop : Population
        Object of type Population which holds population's collective
        representation, validity state and fitness values.
    seed : int (inherited)
        The seed for random numbers generators.
    device : str (inherited)
        Specification of the processing device.
    """
    __name__ = "A-PSO"

    def __init__(self, pi, initializer, mutator, v_clamp=4.0, pop_size=100, seed=0, device="cpu"):
        """Objects' constructor.

        Parameters
        ----------
        pi : Problem
            Optimization problem's instance (PI).
        initializer : function
            The initialization procedure.
        mutator : function
            A function to move solutions across the solve space.
        v_clamp : float (default=4.0)
            Velocity's clamping bound. If not None, 𝒗_𝒑(𝒕)  is not
            clamped; otherwise will be clamped at +/- v_clamp.
        pop_size : int (default=100)
            Population's size.
        seed : int (default=0)
            The seed for random numbers generators.
        device : str (default="cpu")
            Specification of the processing device.
        """
        SPSO.__init__(self, pi, initializer, mutator, v_clamp, pop_size, seed, device)

    def _update(self, it=0, it_max=20, test_elite=False):
        """ Executes step 2) of PSO algorithm

        For the sake of computational efficiency (to allow vectorized
        operations), the whole set of sub-steps of 2) was placed in
        this method. First, for each particle in the swarm, the PSO's
        force generating mechanism is applied, and particle's position
        is updated; after, the particle is evaluated and the respective
        local best is updated. The update of the global best is
        performed or after the update and evaluation of each particle.

        Parameters
        ----------
        it : int (default=0)
            Current iteration. Required to adjust the inertia weight.
        it_max : int (default=20)
            Maximum amount of iterations. Required to adjust the
            inertia weight.
        test_elite : bool (default=False)
            A flag indicating whether the best-so-far solution (𝑖)
            should be evaluated on the test partition (regards SML-OPs).
        """
        for i in range(self.pop_size):
            # 2) 4)
            velocity_i = self.mutator(self.pop.repr_[i], self.pop.velocity[i], self.pop.lBest_repr_[i],
                                      self.best_sol.repr_, it, it_max)
            # Clamps the velocity for particle i
            if self.v_clamp is not None or self.v_clamp > 0.0:
                velocity_i[torch.where(-self.v_clamp < velocity_i, 0, 1).bool()] = -self.v_clamp
                velocity_i[torch.where(velocity_i < self.v_clamp, 0, 1).bool()] = self.v_clamp

            # 2) 5)
            self.pop.repr_[i] += velocity_i

            # 2) 1)
            s_i = Solution(self.pop.repr_[i])
            self.pi.evaluate_sol(s_i)
            self.pop.fit[i] = s_i.fit
            self.pop.valid[i] = s_i.valid

            # 2) 2)
            if self.pi.min_:
                if s_i.fit <= self.pop.lBest_fit[i]:
                    self.pop.lBest_fit[i] = s_i.fit
                    self.pop.lBest_repr_[i] = s_i.repr_.clone()
            else:
                if s_i.fit >= self.pop.lBest_fit[i]:
                    self.pop.lBest_fit[i] = s_i.fit
                    self.pop.lBest_repr_[i] = s_i.repr_.clone()

            # 2) 3)
            self.best_sol = self._get_best(self.best_sol, s_i)._get_copy()
            # Optionally, evaluates the elite on the test set
            if test_elite:
                self.pi.evaluate_sol(self.best_sol, test=True)
