import sys
import time

import torch

from gpol.problems.problem import Problem
from gpol.utils.solution import Solution
from gpol.utils.inductive_programming import _execute_tree, _get_tree_depth


class SSNFusionLogits(Problem):
    """ SS networks' fusion of logits by means of programs' induction.

    This class provides facilities to assess candidate solutions on a
    given Semantics Segmentation (SS) task. It assumes solutions'
    representation as trees of program elements, stored as lists, where
    the terminals stand for the SS networks' output logits (i.e., the
    volume prior predicted classes' generation by means of argmax).
    Under this perspective, a terminal has the following shape:
     [batch size X nº of classes X image's height X images' width]
    The tree represents networks' fusion policy.


    Attributes
    ----------
    sspace : dict
        The solve space composed by:
         - X : torch.Tensor
          The input data. From its shape, we can derive the number
          training instances and input features (the major component
          of the terminal set). Within this context, the dimensionality,
          i.e., the number of input features, is implicit (unlike in
          the case of ConstrainedContinuousFunction or Knapsack, where
          it has to be explicitly defined);
         - y (torch.Tensor): target feature;
         - p_validation double): proportion of training instances used
          for validation;
         - function set: the set of all primitive functions used to
          compose trees;
         - % of constants: proportion of constants in regard to the number
          of input features, (the minor component of the terminal set);
         - range of constants: lower and upper bounds for random constants
          generation;
         - max_init_depth: maximum initialization depth;
         - max_growth_depth: maximum growth depth of trees.
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - any other function with at least two arguments - y_true and y_pred
    min_ : bool
        A flag which defines the purpose of optimization.
    """

    __name__ = "InductiveProgramming"

    def __init__(self, sspace, ffunction, min_=True):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The formal definition of 𝑆 (problem dependent).
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅.
        min_ : bool (default=True)
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)
        self.device = self.sspace["constant_set"][0].device

    def evaluate_pop(self, pop, validation=False):
        # Registers timing to print on the console
        start = time.time()
        # Which data partition to use
        data_loader = "val_loader" if validation else "train_loader"
        # Validates population's representation
        pop.valid = torch.ones(len(pop.representation), dtype=torch.bool)
        # Creates a temporary tensor which for intermediate calculations of fitness values
        fitness_cases = torch.zeros(len(pop.representation), device=self.device, dtype=torch.float64)
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # For each batch in the data loader
            for b, (X_batch, y_batch) in enumerate(self.sspace[data_loader]):
                # Iterates for n_batches training batches, if mode is 'training'
                if data_loader == "train_loader" and b == self.sspace["n_batches"]:
                    break

                # Prints batch's number on the console
                print("BATCH:", b)
                # Loads the batch and places it to the respective processing device
                X_batch, y_batch = X_batch.squeeze(1).to(self.device), y_batch.to(self.device)
                # For the current batch, iterates over the population to evaluate_pop each individual
                for r, (representation, valid) in enumerate(zip(pop.representation, pop.valid)):
                    # Evaluates the solution, if it is valid
                    if valid:
                        # Checks if a tree is made of constants only
                        constant_semantics = True
                        for program_element in representation:
                            if type(program_element) == int:
                                constant_semantics = False
                                break
                        # If the tree is made of constants only, assigns it a 'very bad' fitness
                        if constant_semantics:
                            fitness_cases[r] = torch.tensor([float(sys.maxsize)], device=self.device) if self.minimization else \
                                torch.tensor([0.0], device=self.device)
                            continue
                        else:
                            y_pred = _execute_tree(representation, X_batch)
                            fitness_cases[r] += self.fitness_function(y_true=y_batch, y_pred=y_pred)

                torch.cuda.empty_cache()

        if validation:
            # Assigns solutions' validation fitness. Assumes that all validation set is used for fitness calculation.
            pop.validation_fitness = fitness_cases / (b + 1) * 1.0
            # Prints on the console
            print("> > > TIMING for population's evaluation: ", time.time() - start)
            for r, (rep, fit) in enumerate(zip(pop.representation, pop.validation_fitness)):
                print("> > > ... R: ", r, len(rep), rep, fit)
            print()
        else:
            # Assigns solutions' fitness
            pop.fitness = fitness_cases / self.sspace["n_batches"] * 1.0
            # Prints on the console
            print("> > > TIMING for population's evaluation: ", time.time() - start)
            for r, (rep, fit) in enumerate(zip(pop.representation, pop.fitness)):
                print("> > > ... R: ", r, len(rep), rep, fit)
            print()

    def evaluate_sol(self, solution, validation=True):
        """ Evaluates a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _validate, evaluates
        it by means of 𝑓. If the solution happens to be invalid, then
        it automatically receives a "very bad fitness":  maximum
        possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        solution : Solution
            A candidate solution to evaluate_pop.
        validation : bool
            A flag whether to use training (False) or validation (True) data loader.
        """
        # Registers timing to print on the console
        start = time.time()
        # Which data partition to use
        data_loader = "val_loader" if validation else "train_loader"
        # Creates a temporary variable for fitness calculation
        fitness = 0.0
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # For each batch in the data loader
            for b, (X_batch, y_batch) in enumerate(self.sspace[data_loader]):
                # Loads the batch and places it to the respective processing device
                X_batch, y_batch = X_batch.squeeze(1).to(self.device), y_batch.to(self.device)
                # Checks if a tree is made of constants only
                constant_semantics = True
                for program_element in solution.representation:
                    if type(program_element) == int:
                        constant_semantics = False
                        break
                # If the tree is made of constants only, assigns it a 'very bad' fitness
                if constant_semantics:
                    solution.fitness = torch.tensor([float(sys.maxsize)], device=self.device) if self.minimization else \
                                torch.tensor([0.0], device=self.device)
                    break
                else:
                    y_pred = _execute_tree(solution.representation, X_batch)
                    fitness += self.fitness_function(y_true=y_batch, y_pred=y_pred)

        if validation:
            # Assigns solutions' validation fitness. Assumes that all validation set is used for fitness calculation.
            solution.validation_fitness = fitness / (b+1)*1.0
            # Prints on the console
            print("> > > SOLUTION's validation fitness: ", time.time() - start, solution.validation_fitness, solution.representation)
        else:
            # Assigns solutions' fitness
            solution.fitness = fitness / self.sspace["n_batches"]*1.0
            # Prints on the console
            print("> > > SOLUTION's fitness: ", time.time() - start, solution.fitness, solution.representation)

    def _validate(self, representation):
        """ The procedure to validate a candidate solution.

        This method validates solutions representation given the
        constraints specified in 𝑆. In the context of IP-PO, the
        validity criterion has to do with the depth of the tree-based
        representation of a candidate solution.

        Parameters
        ----------
        representation : torch.tensor
            The representation of a candidate solution to validate.

        Returns
        -------
        bool
            Validity state of input representation.
        """
        if self.sspace["max_growth_depth"] == -1:
            # No depth limit was specified. All solutions are valid.
            return [True]*len(representation)
        else:
            valid = []
            for r in representation:
                valid.append(True if _get_tree_depth(r) <= self.sspace["max_growth_depth"] else False)
            return valid



class SSNFusion():
    """
    Attributes
    ----------
    sspace : dict
        The solve space composed by:
         - X : torch.Tensor
          The input data. From its shape, we can derive the number
          training instances and input features (the major component
          of the terminal set). Within this context, the dimensionality,
          i.e., the number of input features, is implicit (unlike in
          the case of ConstrainedContinuousFunction or Knapsack, where
          it has to be explicitly defined);
         - y (torch.Tensor): target feature;
         - p_validation double): proportion of training instances used
          for validation;
         - function set: the set of all primitive functions used to
          compose trees;
         - % of constants: proportion of constants in regard to the number
          of input features, (the minor component of the terminal set);
         - range of constants: lower and upper bounds for random constants
          generation;
         - max_init_depth: maximum initialization depth;
         - max_growth_depth: maximum growth depth of trees.
    fitness_function : function
        ð‘“ : ð‘† â†’ ð¼ð‘…. Examples of possible fitness functions are:
         - mean absolute error;
         - mean squared error;
         - mean squared logarithmic error;
         - median absolute error;
         - any deepSim function with at least two arguments - y_true and y_pred
    minimization : bool
        A flag which defines the purpose of optimization.
    """
    __name__ = "SSNFusion"

    def __init__(self, sspace, fitness_function, minimization=True):
        """ Object's constructor.

        Parameters
        ----------
        sspace : dict
            The formal definition of ð‘† (problem dependent).
        fitness_function : function
            ð‘“ : ð‘† â†’ ð¼ð‘….
        minimization : bool
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, fitness_function, minimization)
        self.device = self.sspace["constant_set"][0].device

    def evaluate(self, population, validation=False):
        # Registers timing to print on the console
        start = time.time()
        # Which data partition to use
        data_loader = "val_loader" if validation else "train_loader"
        # Validates population's representation
        population.valid = torch.ones(len(population.representation), dtype=torch.bool)
        # Creates a temporary tensor which for intermediate calculations of fitness values
        fitness_cases = torch.zeros(len(population.representation), device=self.device, dtype=torch.float64)
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # For each batch in the data loader
            for b, (X_batch, y_batch) in enumerate(self.sspace[data_loader]):
                # Iterates for n_batches training batches, if mode is 'training'
                if data_loader == "train_loader" and b == self.sspace["n_batches"]:
                    break

                # Prints batch's number on the console
                print("BATCH:", b)
                # Loads the batch and places it to the respective processing device
                X_batch, y_batch = X_batch.squeeze(1).to(self.device), y_batch.to(self.device)
                # For the current batch, iterates over the population to evaluate_pop each individual
                for r, (representation, valid) in enumerate(zip(population.representation, population.valid)):
                    # Evaluates the solution, if it is valid
                    if valid:
                        # Checks if a tree is made of constants only
                        constant_semantics = True
                        for program_element in representation:
                            if type(program_element) == int:
                                constant_semantics = False
                                break
                        # If the tree is made of constants only, assigns it a 'very bad' fitness
                        if constant_semantics:
                            fitness_cases[r] = torch.tensor([float(sys.maxsize)], device=self.device) if self.minimization else \
                                torch.tensor([0.0], device=self.device)
                            continue
                        else:
                            y_pred = _execute_tree(representation, X_batch)
                            fitness_cases[r] += self.fitness_function(y_true=y_batch, y_pred=y_pred)

                torch.cuda.empty_cache()

        if validation:
            # Assigns solutions' validation fitness. Assumes that all validation set is used for fitness calculation.
            population.validation_fitness = fitness_cases / (b+1)*1.0
            # Prints on the console
            print("> > > TIMING for population's evaluation: ", time.time() - start)
            for r, (rep, fit) in enumerate(zip(population.representation, population.validation_fitness)):
                print("> > > ... R: ", r, len(rep), rep, fit)
            print()
        else:
            # Assigns solutions' fitness
            population.fitness = fitness_cases / self.sspace["n_batches"] * 1.0
            # Prints on the console
            print("> > > TIMING for population's evaluation: ", time.time() - start)
            for r, (rep, fit) in enumerate(zip(population.representation, population.fitness)):
                print("> > > ... R: ", r, len(rep), rep, fit)
            print()

    def evaluate_solution(self, solution, validation=True):
        """ Evaluates a candidate solution.

        This method receives a candidate solution from ð‘† and, after
        validating its representation by means of _validate, evaluates
        it by means of ð‘“. If the solution happens to be invalid, then
        it automatically receives a "very bad fitness":  maximum
        possible integer in the case of minimization, zero otherwise.

        Parameters
        ----------
        solution : Solution
            A candidate solution to evaluate_pop.
        validation : bool
            A flag whether to use training (False) or validation (True) data loader.
        """
        # Registers timing to print on the console
        start = time.time()
        # Which data partition to use
        data_loader = "val_loader" if validation else "train_loader"
        # Creates a temporary variable for fitness calculation
        fitness = 0.0
        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # For each batch in the data loader
            for b, (X_batch, y_batch) in enumerate(self.sspace[data_loader]):
                # Loads the batch and places it to the respective processing device
                X_batch, y_batch = X_batch.squeeze(1).to(self.device), y_batch.to(self.device)
                # Checks if a tree is made of constants only
                constant_semantics = True
                for program_element in solution.representation:
                    if type(program_element) == int:
                        constant_semantics = False
                        break
                # If the tree is made of constants only, assigns it a 'very bad' fitness
                if constant_semantics:
                    solution.fitness = torch.tensor([float(sys.maxsize)], device=self.device) if self.minimization else \
                                torch.tensor([0.0], device=self.device)
                    break
                else:
                    y_pred = _execute_tree(solution.representation, X_batch)
                    fitness += self.fitness_function(y_true=y_batch, y_pred=y_pred)

        if validation:
            # Assigns solutions' validation fitness. Assumes that all validation set is used for fitness calculation.
            solution.validation_fitness = fitness / len(self.sspace[data_loader])
            # Prints on the console
            print("> > > SOLUTION's validation fitness: ", time.time() - start, solution.validation_fitness, solution.representation)
        else:
            # Assigns solutions' fitness
            solution.fitness = fitness / self.sspace["n_batches"]
            # Prints on the console
            print("> > > SOLUTION's fitness: ", time.time() - start, solution.fitness, solution.representation)

    def _validate(self, representation):
        """ The procedure to validate a candidate solution.

        This method validates solutions representation given the
        constraints specified in ð‘†. In the context of IP-PO, the
        validity criterion has to do with the depth of the tree-based
        representation of a candidate solution.

        Parameters
        ----------
        representation : torch.tensor
            The representation of a candidate solution to validate.

        Returns
        -------
        bool
            Validity state of input representation.
        """
        if self.sspace["max_growth_depth"] == -1:
            # No depth limit was specified. All solutions are valid.
            return [True]*len(representation)
        else:
            valid = []
            for r in representation:
                valid.append(True if _get_tree_depth(r) <= self.sspace["max_growth_depth"] else False)
            return valid
