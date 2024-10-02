import sys
from joblib import Parallel, delayed

import torch

from gpol.problems.problem import Problem
from gpol.utils.solution import Solution
from gpol.utils.inductive_programming import _execute_tree, _get_tree_depth


class SML(Problem):
    """ Implements SML problem in the scope of IP-OPs

    "Inductive programming (IP) is a special area of automatic 
    programming, covering research from artificial intelligence and 
    programming, which addresses learning of typically declarative 
    (logic or functional) and often recursive programs from incomplete 
    specifications, such as input/output examples or constraints."
        - https://en.wikipedia.org/wiki/Inductive_programming
    
    In the context of Supervised Machine Learning (SML) problems, one
    can define the task of a Genetic Programming (GP) algorithm as the
    program/function induction that identifies the mapping 𝑓 : 𝑆 → 𝐼𝑅
    in the best possible way, generally measured through solutions'
    generalization ability. "The generalization ability (or simply
    generalization) of a model is defined by its performance in data
    other than the training data. In practice, this generalization
    ability is estimated by leaving out of the training data a part of
    the total available data. The data left out of the training data is
    usually referred to as unseen data, testing data, or test data. A
    model that is performing well in unseen data is said to be
    generalizing. However, performance in training and unseen data does
    not always agree."
        - An Exploration of Generalization and Overfitting in Genetic 
          Programming - Standard and Geometric Semantic Approaches, 
          I. Gonçalves (2016).

    In the context of this library, and in this release, GP is mainly
    used to solve SML problems, like regression or classification. As
    such, the solve space for an instance of inductive programming OP,
    is made of labeled input training and unseen data, and GP-specific
    parameters which characterize and bound the solve space (i.e., the
    set of functions and constants, the range of randomly  generated
    constants, the maximum boundaries for initial trees' depth, etc.).

    An instance of this class receives the training and unseen data as the
    instance features of type torch.utils.data.Dataset. Consequently the
    training of a GP system can be performed by batches or by using the whole
    dataset at a time.

    Attributes
    ----------
    sspace : dict
        The solve space of an instance of SML OP is composed by the
        following key-value pairs:
            <"n_dims"> int: the number of input features (a.k.a. input
             dimensions) in the underlying SML problem;
            <"function_set"> list: the set of primitive functions;
            <"constant_set"> torch.Tensor: the set of constants to draw
             terminals from;
            <"p_constants"> float: the probability of generating a
             constant when sampling a terminal;
            <"max_init_depth"> int: maximum trees' depth during the
             initialization;
            <"max_depth"> int: maximum trees' depth during the
             evolution;
            <"n_batches"> int: number of batches to use when evaluating
             solutions (more than one can be used).
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - etc.
    min_ : bool
        A flag which defines the purpose of optimization.
    dl_train : torch.utils.data.DataLoader
        Train data-loader.
    dl_test : torch.utils.data.DataLoader
        Test data-loader.
    n_jobs : int
        Number of jobs to run in parallel when executing trees.
    device : str
        Specification of the processing device.
    """
    __name__ = "IP-SML"

    def __init__(self, sspace, ffunction, dl_train, dl_test=None, min_=True, n_jobs=1):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The solve space of an instance of SML OP is composed by the
            following key-value pairs:
                <"n_dims"> int: the number of input features (a.k.a.
                 input dimensions) in the underlying SML problem;
                <"function_set"> list: the set of primitive functions;
                <"constant_set"> torch.Tensor: the set of constants to
                 draw terminals from;
                <"p_constants"> float: the probability of generating a
                 constant when sampling a terminal;
                <"max_init_depth"> int: maximum trees' depth during the
                 initialization;
                <"max_depth"> int: maximum trees' depth during the
                 evolution;
                <"n_batches"> int: number of batches to use when
                 evaluating solutions (more than one can be used).
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅.
        dl_train : torch.utils.data.DataLoader
            DataLoader for the training set.
        dl_test : torch.utils.data.DataLoader
            DataLoader for the testing set.
        n_jobs : int (default=1)
            Number of parallel processes used to execute the trees.
        min_ : bool
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.n_jobs = n_jobs
        # Infers processing device from the constants' set
        self.device = self.sspace["constant_set"].device

    def evaluate_pop(self, pop):
        """ Evaluates a population of candidate solutions.

        This method receives a population of candidate solutions from 𝑆 and, after
        validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        pop : Population
            population candidate solution to evaluate_pop.
        """
        # Validates population's representation
        pop.valid = self._is_feasible_pop(pop.repr_)
            
        # Assigns default fitness-cases 
        pop.fit = torch.zeros(len(pop), device=self.device)

        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # Iterates fitness calculation for n_batches
            for b, batch in enumerate(self.dl_train):
                # Brakes the loop when b equals n_batches
                if b == self.sspace["n_batches"]:
                    break

                # Moves the data to the underlying processing device
                X, y = batch[0].to(self.device), batch[1].to(self.device)

                # For the current batch, iterates over the population to evaluate_pop each individual
                y_pred = Parallel(n_jobs=self.n_jobs)(delayed(_execute_tree)(repr_, X) for repr_ in pop.repr_)
                pop.fit += self.ffunction(y_true=y, y_pred=torch.stack(y_pred))

                # Releases all unoccupied cached memory
                torch.cuda.empty_cache()

        # Assigns the training fitness values to valid solutions in the population
        pop.fit[pop.valid] /= self.sspace["n_batches"]
        # Assigns the default fitness to invalid solutions in the population
        pop_invalid = ~torch.tensor(pop.valid)
        if any(pop_invalid):
            pop.fit[pop_invalid] = torch.ones(pop_invalid.sum(), device=self.device)*sys.maxsize if self.min_ else \
                torch.zeros(pop_invalid.sum(), device=self.device)

    def evaluate_sol(self, sol, test=False):
        """ Evaluate a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero
        otherwise.

        Parameters
        ----------
        sol : Solution
            A candidate solution from the solve space.
        test : bool
            A flag which defines which data partition to use when
            evaluating the solution.
        """
        # Validates solution's representation
        sol.valid = self._is_feasible_sol(sol.repr_)

        # Evaluates solution, if it is valid
        if sol.valid:
            # Chooses which data partition to use
            data_loader = self.dl_test if test else self.dl_train

            # Temporarily sets all the requires_grad flag to False
            with torch.no_grad():
                # Creates a container to accumulate fitness across different batches
                fit = 0.0
                # Iterates fitness calculation for n_batches
                for b, batch in enumerate(data_loader):
                    # Brakes the loop when b equals n_batches
                    if b == self.sspace["n_batches"]:
                        break

                    # Moves the data to the underlying processing device
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    # Computes the semantics
                    y_pred = _execute_tree(sol.repr_, X)
                    # Computes the fitness of a solution the current batch
                    fit += self.ffunction(y_true=y, y_pred=y_pred)

                    # Releases all unoccupied cached memory
                    torch.cuda.empty_cache()

            # Assigns the training fitness values to the solution, according to the partition type
            fit /= self.sspace["n_batches"]
            if test:
                sol.test_fit = fit
            else:
                sol.fit = fit
        else:
            self._set_bad_fit_sol(sol, test, self.device)

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints.

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any). In the context of IP-OP, the feasibility relates
        with the maximum allowed depth of the tree representing a
        candidate solution.

        Parameters
        ----------
        repr_ : list
            LISP-based representation of a candidate solution.

        Returns
        -------
        bool
            Representation's feasibility state.
        """
        if "max_depth" not in self.sspace or self.sspace["max_depth"] == -1:
            # No depth limit was specified, so the solution is valid.
            return True
        else:
            if _get_tree_depth(repr_) <= self.sspace["max_depth"]:
                return True
            else:
                return False

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints.

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.
        In the context of IP-OP, the feasibility relates with the
        maximum allowed depth of the tree representing a candidate
        solution.

        Parameters
        ----------
        repr_ : list
            A list of LISP-based representations of a set of candidate
            solutions.

        Returns
        -------
        list
            Representations' feasibility state.
        """
        if "max_depth" not in self.sspace or self.sspace["max_depth"] == -1:
            # No depth limit was specified, thus all the solutions are assumed to be valid
            return [True]*len(repr_)
        else:
            return [_get_tree_depth(t) <= self.sspace["max_depth"] for t in repr_]


class SMLGS(Problem):
    """ Implements inductive programming OP.

    "Inductive programming (IP) is a special area of automatic
    programming, covering research from artificial intelligence and
    programming, which addresses learning of typically declarative
    (logic or functional) and often recursive programs from incomplete
    specifications, such as input/output examples or constraints."
        - https://en.wikipedia.org/wiki/Inductive_programming

    In the context of Machine Learning (ML) OPs, one can define the
    task of Genetic Programming (GP) as of program/function induction
    that identifies the mapping 𝑓 : 𝑆 → 𝐼𝑅 in the best possible way,
    generally measured through tge generalization ability.
    "The generalization ability (or simply generalization) of a model
    is dened by its performance in data other than the training data.
    In practice, this generalization ability is estimated by leaving
    out of the training data a part of the total available data. The
    data left out of the training data is usually referred to as
    unseen data, testing data, or test data. A model that is
    performing well in unseen data is said to be generalizing. However,
    performance in training and unseen data does not always agree."
        - An Exploration of Generalization and Overfitting in Genetic
          Programming - Standard and Geometric Semantic Approaches,
          I. Gonçalves (2016).

    In the context of this library, GP is mainly used to solve
    Supervised Machine Learning (SML) tasks, like Regression or
    Classification. As such, the solve space for an instance of
    inductive programming OP, in the context of Machine Learning (ML)
    problem solving, is made of labeled input training and unseen data,
    and GP-specific parameters which characterize and delimit the
    solve space (i.e., the set of functions and constants, the range of
    randomly  generated constants, the maximum boundaries for initial
    trees' depth, etc.).

    This class implements another OP type and is subject to the
    following restrictions:
     - the training and unseen data are provided within the solve
       space as objects of type torch.utils.data.Dataset;
     - the training of GP system can be performed by batches or by
       using the whole dataset at a time.

    Attributes
    ----------
    sspace : dict
        The solve space composed by:
         - batch_size : int
         - function set: the set of all primitive functions used to
          compose trees;
         - % of constants: proportion of constants in regard to the number
          of input features, (the minor component of the terminal set);
         - range of constants: lower and upper bounds for random constants
          generation;
         - max_init_depth: maximum initialization depth;
         - max_growth_depth: maximum growth depth of trees.
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
    test_indices : torch.Tensor
        Indices representing the testing partition of the data.
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - etc.
    min_ : bool
        A flag which defines the purpose of optimization.
    device : str
        Specification of the processing device. Inferred from the
        solve-space's definition.
    """
    __name__ = "IPOP_GS"

    def __init__(self, sspace, ffunction, X, y, train_indices, test_indices, batch_size=None, min_=True):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The formal definition of 𝑆 (problem dependent).
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅.
        X : torch.Tensor
            A tensor representing the input samples. Internally, it
            will be converted to dtype=torch.float. It is assumed that
            the object was already allocated on the proper processing
            device.
        y : torch.Tensor
            A tensor representing the The target values. Internally, it
            will be converted to dtype=torch.float. It is assumed that
            the object was already allocated on the proper processing
            device.
        min_ : bool (default=True)
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        # Infers the processing device
        self.device = self.sspace["constant_set"].device
        # Splits the data
        self.train_indices, self.test_indices = train_indices, test_indices

    def evaluate_pop(self, pop):
        """ Evaluates a population of candidate solutions.

        This method receives a population of candidate solutions from 𝑆 and, after
        validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        pop : Population
            population candidate solution to evaluate_pop.
        """
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # Assigns default validity state: with GSOs it is True, by default, for all the individuals
            pop.valid = torch.ones(len(pop), dtype=torch.bool, device=self.device)
            # Computes population's fitness: with or without batch training
            if self.batch_size:
                # Randomly generates indices representing a random batch
                batch_indices = self.train_indices[torch.randperm(len(self.train_indices))[0:self.batch_size]]
                # Obtains tree's prediction
                y_pred = pop.repr_[:, batch_indices]
                # Computes the solutions' fitness cases
                pop.fit = self.ffunction(y_true=self.y[batch_indices], y_pred=y_pred)
            else:
                # Obtains tree's prediction
                y_pred = pop.repr_[:, self.train_indices]
                # Computes the solutions' fitness cases
                pop.fit = self.ffunction(y_true=self.y[self.train_indices], y_pred=y_pred)

        # Releases all unoccupied cached memory
        torch.cuda.empty_cache()

    def evaluate_sol(self, sol, test=False):
        """ Evaluate a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero
        otherwise.

        Parameters
        ----------
        sol : Solution
            A candidate solution from the solve space.
        test : bool
            A flag which defines which data partition to use when
            evaluating the solution.
        """
        # Assigns default validity state: with GSOs it is true, by default
        sol.valid = True

        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # Selects partition's indexes for the solution's semantics
            idxs = self.test_indices if test else self.train_indices
            # Selects target semantics based on the partition
            if self.batch_size:
                # Randomly generates indices representing a random batch
                idxs = idxs[torch.randperm(len(idxs))[0:self.batch_size]]
                # Obtains tree's prediction
                y_pred = sol.repr_[idxs]
            else:
                y_pred = sol.repr_[idxs]

            # Computes the solution's fitness
            if test:
                sol.test_fit = self.ffunction(y_true=self.y[idxs], y_pred=y_pred)
            else:
                sol.fit = self.ffunction(y_true=self.y[idxs], y_pred=y_pred)

        # Releases all unoccupied cached memory
        torch.cuda.empty_cache()
