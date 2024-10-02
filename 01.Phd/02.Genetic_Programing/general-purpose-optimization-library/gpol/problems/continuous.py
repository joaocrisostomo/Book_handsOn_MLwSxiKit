import torch

from gpol.problems.problem import Problem


class Box(Problem):
    """ Implements a constrained continuous OP.

    Continuous OPs are mainly presented in two variants: unconstrained 
    or constrained. The unconstrained OPs do not impose any explicit
    spacial constraints on the candidate solution's feasibility; in
    practice, solution's representation is constrained by the underlying
    data type. Contrarily, the constrained OPs impose explicit spacial
    constraints on the solve space.
    
    This class implements a simplistic variant of the constrained OPs,
    where the parameters can take any real number within a given range 
    of values - the box. Each box's dimensions can have equal (regular) 
    or unequal (irregular) constraints.
    
    When solving constrained continuous OPs, some researchers bound
    solutions to prevent the solve in infeasible regions. The bounding
    mechanism typically consists of a randomized reinitialization of
    the solution on the outlying dimension. For this reason, this
    library implements solutions' bounding, as an optional feature.

    Attributes:
    ----------
    sspace : dict
        The solve space (𝑆) of an instance of Box OP consisting of the
        following key-value pairs:
            <"n_dims"> int: the number of problem's dimensions;
            <"constraints"> torch.Tensor: the constraints on 𝑆.
    ffunction : function
        𝑓 : 𝑆 → 𝐼𝑅 (i.e., the fitness function). Examples of possible
        fitness functions are Ackley, Rastrigin, Rosenbrock, etc.
    bound : bool
        A flag that states whether to apply (True) or not (False) the
        bounding mechanism, consisting of a randomized reinitialization
        of the solution(s) on the outlying dimension.
    min_ : bool
        A flag which defines optimization's purpose. If it is True,
        then the OP is a minimization problem; maximization otherwise.
    """
    __name__ = "Box"
    
    def __init__(self, sspace, ffunction, bound=False, min_=True):
        """ Object's constructor

        Parameters
        ----------
        sspace : dict
            The formal definition of the 𝑆 consisting of the following
            key-value pairs:
                <"n_dims"> int: the number of problem's dimensions;
                <"constraints"> torch.Tensor: the hyper-cube of 𝑆.
        ffunction : function
            𝑓 : 𝑆 → 𝐼𝑅. Examples of possible fitness functions are Ackley,
            Rastrigin, Rosenbrock, etc.
        bound : bool (default=False)
            A flag that states whether to apply (True) or not (False) the
            bounding mechanism, consisting of a randomized reinitialization
            of the solution(s) on the outlying dimension (implemented in
            validation methods).
        min_ : bool
            A flag which defines optimization's purpose. If it's value
            is True, then the OP is a minimization problem; otherwise
            it is a maximization problem.
        """
        Problem.__init__(self, sspace, ffunction, min_)
        self.bound = bound        

    def evaluate_sol(self, sol):
        """ Evaluates a candidate solution

        This method receives a candidate solution from 𝑆 and, after
        validating its representation by means of _is_feasible_sol,
        evaluates it by means of 𝑓. If the solution happens to be
        invalid, it automatically receives a "very bad fitness":
        maximum possible value in the case of minimization, zero
        otherwise.

        In the context of continuous OPs, a solution is represented
        by one dimensional tensor of floats, where each successive
        value regards a different dimension.

        Parameters
        ----------
        sol : Solution
            A candidate solution.
        """
        sol.valid = self._is_feasible_sol(sol.repr_)

        if sol.valid:
            sol.fit = self.ffunction(sol.repr_)
        else:
            self._set_bad_fit_sol(sol)

    def evaluate_pop(self, pop):
        """ Evaluates a population of candidate solutions

        This method receives a population of solutions and, after
        validating its representation by means of _is_feasible_pop,
        evaluates it by means of 𝑓. If some solutions happen to be
        unfeasible, they receive a "very bad fitness": maximum possible
        value in the case of minimization, zero otherwise.

        In the context of Continuous OP, population's representation is
        given by a two dimensional tensor of floats where the first
        dimension enumerates different individuals, while the second
        the respective values across several dimensions.

        Parameters
        ----------
        pop : Population
            The object which holds population's representation and other
            important attributes (e.g. fitness cases, feasibility
            states, etc.).
        """
        # Validates population's representation
        pop.valid = self._is_feasible_pop(pop.repr_)
        # Assigns default fitness values
        self._set_bad_fit_pop(pop, device=pop.repr_.device)

        # Temporarily sets all the requires_grad flag to False
        with torch.no_grad():
            # Computes fitness of population's valid individuals
            pop.fit[pop.valid] = self.ffunction(pop.repr_[pop.valid])
            # Releases all unoccupied cached memory
            torch.cuda.empty_cache()

    def _is_feasible_sol(self, repr_):
        """ Assesses solution's feasibility under 𝑆's constraints

        Assesses solution's feasibility after constraints specified
        in 𝑆 (if any).

        Parameters
        ----------
        repr_ : torch.Tensor
            Representation of a candidate solution.

        Returns
        -------
        bool
            Representations's feasibility state.
        """
        valid_dims = torch.logical_and(self.sspace["constraints"][0] <= repr_, self.sspace["constraints"][1] >= repr_)
        if len(self.sspace["constraints"].shape) > 1:  # when the constraints is irregular
            if valid_dims.sum() == len(repr_):
                return True  # returns True when all the dimensions are within the constraints
            else:
                if self.bound:
                    invalid_dims = ~valid_dims
                    replacement = torch.distributions.Uniform(self.sspace["constraints"][0][invalid_dims],
                                                              self.sspace["constraints"][1][invalid_dims]).sample()

                    repr_[invalid_dims] = replacement.to(repr_.device)
                    return True
                else:
                    return False
        else:  # when the constraints is regular
            if valid_dims.sum() != len(repr_):  # when the solution is not valid
                if self.bound:  # when one wants to apply the bounding mechanism
                    invalid_dims = ~valid_dims
                    n = invalid_dims.sum().item()
                    repr_[invalid_dims] = torch.FloatTensor(n).uniform_(self.sspace["constraints"][0],
                                                                        self.sspace["constraints"][1]).to(repr_.device)
                    return True
                else:  # no bounding
                    return False
            else:  # the solution is valid
                return True

    def _is_feasible_pop(self, repr_):
        """ Assesses population's feasibility under 𝑆's constraints

        Assesses population's feasibility after constraints specified
        in 𝑆 (if any). This method was particularly designed to include
        more efficient assessment procedure for a set of solutions.

        Parameters
        ----------
        repr_ : object
            Candidate solutions's collective representation.

        Returns
        -------
        torch.Tensor
            Representations' feasibility state.
        """
        valid_dims = torch.logical_and(self.sspace["constraints"][0] <= repr_, self.sspace["constraints"][1] >= repr_)
        invalid_dims = ~valid_dims
        if len(self.sspace["constraints"].shape) > 1:  # when the constraints is irregular
            if invalid_dims.any():  # if there is any invalid solution
                if self.bound:  # if the invalid solutions are to be bounded
                    replacement = []
                    for di, d in enumerate(invalid_dims):
                        if d.any():
                            replacement.extend(list(map(lambda y, z: torch.FloatTensor(1).uniform_(y, z),
                                                        self.sspace["constraints"][0][invalid_dims[di]],
                                                        self.sspace["constraints"][1][invalid_dims[di]])))
                    repr_[invalid_dims] = torch.tensor(replacement, device=repr_.device)
                    return torch.ones(len(repr_), dtype=torch.bool, device=repr_.device)
                else:
                    return ~invalid_dims.any(1)
            else:  # if there are no invalid solutions
                return torch.ones(len(repr_), dtype=torch.bool, device=repr_.device)
        else:  # when the constraints is regular
            if invalid_dims.any():
                if self.bound:
                    n = invalid_dims.sum().item()
                    repr_[invalid_dims] = torch.FloatTensor(n).uniform_(self.sspace["constraints"][0],
                                                                        self.sspace["constraints"][1]).to(repr_.device)
                    return torch.ones(len(repr_), dtype=torch.bool, device=repr_.device)
                else:
                    return ~invalid_dims.any(1)
            else:
                return torch.ones(len(repr_), dtype=torch.bool, device=repr_.device)
