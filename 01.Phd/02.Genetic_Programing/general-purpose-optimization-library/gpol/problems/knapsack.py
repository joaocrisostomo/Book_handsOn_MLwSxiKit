import torch

from gpol.problems.problem import Problem


class Knapsack01(Problem):
    """ Implements 0-1 Knapsack OP

    Given a set of items, each characterized by a weight and a value,
    solving an instance of the Knapsack problem consists of determining 
    which items to include in the knapsack so that the total weight 
    is less than or equal to a given limit and the total value is as 
    large as possible.

    The code contained in this class implements the so-called 0-1
    Knapsack problem. In scope of this OP, a candidate solution is
    represented as a 1D tensor of zeros and ones which length equals
    the dimensionality of the solve space (i.e., the number of items);
    each index of a tensor corresponds to a given item in. In such a
    way, the items to be included are the ones for which the tensor
    holds a value 1; this means that, in 0-1 Knapsack OP, a given item
    can be included only once. The evaluation of a candidate solution
    consists of:
        1) validation of a candidate solution's weight (if it fits the
         maximum capacity of the Knapsack);
        2) calculation of the total value of a solution by summing
        value of those items which are mapped with a 1 in candidate
        solution's tensor.

    Note that the tensors representing candidate solutions, although
    holding 0-1 values, are stored as floats for operational reasons.
    For example, to allow torch.matmul and torch.dot functions to
    be applied without additional casts.

    Attributes:
    ----------
    search_space : dict
        The solve space (𝑆) of an instance of a 0-1 knapsack problem.
        It consists of the following key-value pairs:
            <"n_dims"> int: the total number of unique items in 𝑆;
            <"capacity"> float: knapsack's capacity;
            <"weights"> torch.Tensor: items' weights;
            <"values"> torch.Tensor: items' values.
    fit_function : function
        𝑓 : 𝑆 → 𝐼𝑅.
    min_ : bool
        A flag which defines optimization's purpose. If it's value is 
        True, then the OP is a minimization problem; otherwise it is
        a maximization problem.
    """
    __name__ = "Knapsack01"

    def __init__(self, sspace, ffunction, min_=False):
        """ Object's constructor

        Following the main purpose of an instance of a Knapsack OP
        (defined on a 1D binary tensor), the constructor takes as the
        parameters the maximum solve space definition, composed by the
        maximum capacity of the Knapsack, the item set and its length
        (the dimensionality), the fitness function to evaluate_pop the
        candidate solutions, and a flag representing the purpose of the
        optimization problem (OP).

        Parameters
        ----------
        sspace : dict
            The solve space composed by the following key-value pairs:
                <"n_dims"> int: the total number of unique items in 𝑆;
                <"capacity"> float: knapsack's capacity;
                <"weights"> torch.Tensor: items' weights;
                <"values"> torch.Tensor: items' values;
                <"bounds"> torch.Tensor: a 2xn_dims tensor of integers
                    holding the minimum and the maximum number of items'
                    copies.
        ffunction  : function
            𝑓.
        min_ : bool (default=True)
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)

    def evaluate_sol(self, sol):
        """ Evaluates a candidate solution

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, it automatically receives a "very bad fitness":
        maximum possible value in the case of minimization, zero
        otherwise.

        In the context of Knapsack01 OP, solutions' are represented
        by one dimensional tensors of integers (either zeros or ones),
        stored as floats for operational reasons, where each successive
        value regards a different dimension.

        Note that for problems like the Knapsack01 (pure OPs), no test
        partition is required.

        Parameters
        ----------
        sol : Solution
            A candidate solution to be evaluated.
        """
        # 1)
        sol.valid, sol.weight = self._is_feasible_sol(sol.repr_)
        # 2)
        if sol.valid:
            sol.fit = self.ffunction(sol.repr_, self.sspace["values"])
        else:
            self._set_bad_fit_sol(sol)

    def evaluate_pop(self, pop):
        """  Evaluates a population of candidate solutions

        This method receives a population of solutions from 𝑆 and,
        after validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If some solutions happen to be
        invalid, these automatically receive a "very bad fitness":
        maximum possible value in the case of minimization, zero
        otherwise.

        In the context of Knapsack01 OP, populations' are represented
        by two dimensional tensors of integers (either zeros or ones),
        stored as floats for operational reasons. The first dimension
        stands for the individual candidate solutions, whereas the
        second represents their values across several dimensions.

        Note that for problems like the Knapsack01 (pure OPs), no test
        partition is required.

        Parameters
        ----------
        pop : Population
            The object which holds population's representation and
            other important attributes (e.g. fitness cases, validity
            states, etc.).
        """
        # Validates population's representation
        pop.valid, pop.weight = self._is_feasible_pop(pop.repr_)
        # Assigns default fitness values
        self._set_bad_fit_pop(pop, device=pop.repr_.device)
        # Compute the fitness only for the valid solutions
        pop.fit[pop.valid] = self.ffunction(pop.repr_[pop.valid], self.sspace["values"])

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints.

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any). More specifically, the method computes
        solution's total weight as the dot product between its
        representation and the respective weights.

        Parameters
        ----------
        repr_ : torch.Tensor
            Representation of a candidate solution.

        Returns
        -------
        bool
            Representations's feasibility state.
        torch.Tensor
            Total weight of solution's representation.
        """
        # Compute representation's weight
        weight = torch.dot(repr_, self.sspace["weights"])

        return (weight <= self.sspace["capacity"]), weight

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints.

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.
        More specifically, the method computes solutions' total weights
        as the dot product between their representations and the
        respective weights.

        Parameters
        ----------
        repr_ : torch.Tensor
            Candidate solutions's collective representation.

        Returns
        -------
        valid : torch.Tensor
            Representations' feasibility state.
        weight : torch.Tensor
            Total weights of solutions' representations.
        """
        # Compute the weight of solutions' representations
        weight = torch.matmul(repr_, self.sspace["weights"])
        # Assess representations' validity
        valid = weight <= self.sspace["capacity"]

        return valid, weight


class KnapsackBounded(Knapsack01):
    """ Implements Bounded Knapsack OP

    Attributes:
    ----------
    search_space : dict
        The solve space (𝑆) of an instance of a 0-1 knapsack problem.
        It consists of the following key-value pairs:
            "capacity": knapsack's capacity;
            "weights": items' weights;
            "values": items' values;
            "bounds": 2xd tensor of integer values (...)
    fit_function : function
        𝑓 : 𝑆 → 𝐼𝑅.
    min_ : bool
        A flag which defines optimization's purpose. If it's value is
        True, then the OP is a minimization problem; otherwise it is
        a maximization problem.
    """
    __name__ = "KnapsackBounded"

    def __init__(self, sspace, ffunction, min_=False):
        """ Object's constructor

        Parameters
        ----------
        sspace : dict
            The solve space composed by Knapsack's capacity, the item set
             and, implicitly, its length (the dimensionality).
        ffunction  : function
            𝑓.
        min_ : bool
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints.

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any). More specifically, the method computes
        solution's total weight as the dot product between its
        representation and the respective weights.

        Parameters
        ----------
        repr_ : torch.Tensor
            Representation of a candidate solution.

        Returns
        -------
        bool
            Representations's feasibility state.
        torch.Tensor
            Total weight of solution's representation.
        """
        # Verifies the constraint on the min/max number of items
        valid_dims = torch.logical_and(self.sspace["bounds"][0] <= repr_, self.sspace["bounds"][1] >= repr_)
        # Computes representation's weight
        weight = torch.dot(repr_, self.sspace["weights"])
        # Computes representation's validity
        valid = valid_dims.sum() == len(repr_) and weight <= self.sspace["capacity"]

        return valid, weight

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints.

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.
        More specifically, the method computes solutions' total weights
        as the dot product between their representations and the
        respective weights.

        Parameters
        ----------
        repr_ : torch.Tensor
            Candidate solutions's collective representation.

        Returns
        -------
        valid : torch.Tensor
            Representations' feasibility state.
        weight : torch.Tensor
            Total weights of solutions' representations.
        """
        # Verifies the constraint on the min/max number of items
        valid_dims = torch.logical_and(self.sspace["bounds"][0] <= repr_, self.sspace["bounds"][1] >= repr_)
        # Compute the weight of solutions' representations
        weight = torch.matmul(repr_, self.sspace["weights"])
        # Computes representation's validity
        valid = torch.logical_and(valid_dims.sum(1) == len(repr_[0]), weight <= self.sspace["capacity"])

        return valid, weight
