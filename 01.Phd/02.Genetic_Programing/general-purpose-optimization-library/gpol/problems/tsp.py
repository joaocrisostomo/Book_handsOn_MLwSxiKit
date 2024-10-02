import torch

from gpol.problems.problem import Problem


class TSP(Problem):
    __name__ = "TSP"

    """ Implements TSP OP.

    Attributes:
    ----------
    search_space : dict
        The solve space composed by Knapsack's capacity, the item set
         and, implicitly, its length (the dimensionality).
    fit_function : function
        𝑓 : 𝑆 → 𝐼𝑅.
    min_ : bool
        A flag which defines the purpose of optimization.
    min_fit : torch.Tensor
        Minimum possible fitness a solution can have (used for 
        penalization of invalid solutions).
    max_fit : torch.Tensor
        Maximum possible fitness a solution can have (used for 
        penalization of invalid solutions).
    """

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
            The solve space composed by Knapsack's capacity, the item set
             and, implicitly, its length (the dimensionality).
        ffunction  : function
            𝑓.
        min_ : bool
            A flag which defines the purpose of optimization.
        """
        Problem.__init__(self, sspace, ffunction, min_)
        other_cities = list(range(len(self.sspace["distances"])))
        other_cities.remove(self.sspace["origin"])
        self.sspace["other_cities"] = torch.tensor(other_cities, device=self.sspace["distances"].device)

    def evaluate_sol(self, sol):
        """ Evaluates a candidate solution.

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, then it automatically receives a "very bad fitness":
        maximum possible integer in the case of minimization, zero
        otherwise. The evaluation of a candidate solution happens in
        two steps:
            1) validation according to the restrictions specified in 𝑆;
            2) given the validity state of a solution, if it is valid,
             evaluate_pop it using 𝑓. If it is not valid, assign it a
             very high or a very low fitness value, according to the
             purpose of the OP.

        Notice that when considering such a set of problems like
        Knapsack, pure optimization problems (OP), no test partition is
        required.

        Parameters
        ----------
        sol : Solution
            A candidate solution to be evaluated.
        """
        sol.valid = True
        sol.fit = self.ffunction(self.sspace["distances"], self.sspace["origin"], sol.repr_)

    def evaluate_pop(self, pop):
        """  Evaluates a population (set) of candidate-solutions.

        Given the logic embedded in this framework and the fact that,
        formally, an instance of OP is a pair of objects (𝑆, 𝑓), one of
        the problem instance's (PI's) tasks is to evaluate_pop candidate
        solutions' fitness. This method allows for efficient evaluation
        of a set of solutions at a call; this set is encapsulated into
        a special object of type Population.

        Parameters
        ----------
        pop : Population
            The object which holds population's representation and
            other important attributes, such as: the fitness cases,
            the validity states, etc.
        """
        pop.valid = torch.ones(len(pop.repr_), dtype=torch.bool, device=pop.repr_.device)
        pop.fit = self.ffunction(self.sspace["distances"], self.sspace["origin"], pop.repr_)
