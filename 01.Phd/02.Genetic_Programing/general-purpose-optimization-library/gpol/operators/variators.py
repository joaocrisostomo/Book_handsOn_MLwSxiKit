""" Variation operators "move" the candidate solutions across S
The module `gpol.variators` contains some relevant variation operators
(variators) used to "move" the candidate solutions across the solve
space in between the iterations. Given the fact this library supports
different types of iterative solve algorithms (ISAs), the module
contains a collection of variators suitable for every kind of ISA
implemented in this library.
"""

import copy
import random

import torch
import numpy as np

from gpol.algorithms.swarm_intelligence import APSO
from gpol.utils.inductive_programming import tanh1, lf1, add2, sub2, mul2, _execute_tree, get_subtree, _Function


# +++++++++++++++++++++++++++ Inductive Programming
def swap_xo(p1, p2):
    """ Implements the swap crossover

    The swap crossover (a.k.a. standard GP's crossover) consists of
    exchanging (swapping) parents' two randomly selected subtrees.

    Parameters
    ----------
    p1 : list
        Representation of the first parent.
    p2 : list
        Representation of the second parent.

    Returns
    -------
    list, list
        Tuple of two lists, each representing an offspring obtained
        from swapping two randomly selected sub-trees in the parents.
    """
    # Selects start and end indexes of the first parent's subtree
    p1_start, p1_end = get_subtree(p1)
    # Selects start and end indexes of the second parent's subtree
    p2_start, p2_end = get_subtree(p2)

    return p1[:p1_start] + p2[p2_start:p2_end] + p1[p1_end:], p2[:p2_start] + p1[p1_start:p1_end] + p2[p2_end:]


def prm_gs_xo(initializer, device):
    """ Implements the geometric semantic crossover (GSC)

    This function is used to provide the gs_xo (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator and the processing device. The former is necessary to
    generate a random tree that is required to perform the crossover
    itself, whereas the latter is used to create a tensor that holds
    a single value (-1) and store it in the outer scope of gs_xo (this
    is done to avoid allocating it on the GPU at every function's call).

    Parameters
    ----------
    initializer : function
        Parametrized initialization function to generate random trees.
    device : str
        Processing device.

    Returns
    -------
    gs_xo : function
        A function which returns two offsprings after applying the GSC
        on the parents' representation.
    """
    c1 = torch.Tensor([1.0]).to(device)

    def gs_xo(p1, p2):
        """ Implements the geometric semantic crossover

        The GSO corresponds to the geometric crossover in the semantic
        space. This function stores individuals' representations (the
        trees) in memory.

        Parameters
        ----------
        p1 : list
            Representation of the first parent.
        p2 : list
            Representation of the second parent.

        Returns
        -------
        list, list
            Tuple of two lists, each representing for an offspring obtained
            from applying the GSC on parents' representation.
        """
        rt = [lf1] + initializer()
        # Performs GSC on trees and returns the result
        return [add2, mul2] + rt + p1 + [mul2, sub2, c1] + rt + p2, \
               [add2, mul2] + rt + p2 + [mul2, sub2, c1] + rt + p1

    return gs_xo


def prm_efficient_gs_xo(X, initializer):
    """ Implements the an efficient variant of GSC

    This function is used to provide the efficient_gs_xo (inner
    function) the necessary environment (the outer scope) - the
    input data and the random trees' generator. The former is necessary
    to generate a random tree that is required to latter the crossover
    itself, whereas the former is used the latter is necessary to
    execute the aforementioned random tree and store its semantics
    (along with some other features).

    Moreover, this function creates a tensor that holds a single value
    (-1) and store it in the outer scope of gs_xo (this is done to
    avoid allocating it on the GPU at every function's call).

    Parameters
    ----------
    X : torch.tensor
        The input data.
    initializer : function
        Initialization function. Used to generate random trees.

    Returns
    -------
    efficient_gs_xo : function
        A function which returns offsprings' semantics (and some other
        important features), after applying the GSC on the parents'
        representation.
    """
    c1 = torch.tensor([1.0], device=X.device)

    def efficient_gs_xo(p1, p2):
        """ Implements the an efficient variant of GSC

        Implements an efficient variant of GSC that acts on solutions'
        semantics instead of trees' structure. That is, the trees are
        never stored in computers' memory, only one random tree is
        temporarily generated at each function call to allow the
        calculations to happen (its output and other features are
        stored).
         For more details, consult "A new implementation of geometric
        semantic GP and its application to problems in pharmacokinetics"
        by L. Vanneschi et at. (2013).

        Parameters
        ----------
        p1 : list
            Representation of the first parent.
        p2 : list
            Representation of the second parent.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Resulting offsprings' semantics.
        list
            Random tree generated to perform the GSC.
        """
        # Creates a random tree (bounded in [0, 1])
        rt = [lf1] + initializer()
        # Executes the tree to obtain random tree's semantics on X
        rt_s = _execute_tree(rt, X)
        # Performs GSC on semantics and returns parent's semantics and the random tree
        return rt_s * p1 + (c1 - rt_s) * p2, rt_s * p2 + (c1 - rt_s) * p1, rt

    return efficient_gs_xo


def hoist_mtn(repr_):
    """ Implements the hoist mutation

    The hoist mutation selects a random subtree R from solution's
    representation and replaces it with a random subtree R' taken
    from itself, i.e., a random subtree R' is selected from R and
    replaces it in the representation (it is 'hoisted').

    Parameters
    ----------
    repr_ : list
        Parent's representation.

    Returns
    -------
    list
        The offspring obtained from replacing a randomly selected
        subtree in the parent by a random tree.
    """
    # Get a subtree (R)
    start, end = get_subtree(repr_)
    subtree = repr_[start:end]
    # Get a subtree of the subtree to hoist (R')
    sub_start, sub_end = get_subtree(subtree)
    hoist = subtree[sub_start:sub_end]
    # Returns the result as lists' concatenation
    return repr_[:start] + hoist + repr_[end:]


def prm_point_mtn(sspace, prob):
    """ Implements the point mutation

    This function is used to provide the point_mtn (inner function)
    with the necessary environment (the outer scope) - the solve
    space (𝑆), necessary to for the mutation function to access the
    set of terminals and functions.

    Parameters
    ----------
    sspace : dict
        The formal definition of the 𝑆. For GP-based algorithms, it
        contains the set of constants, functions and terminals used
        to perform point mutation.
    prob : float
        Probability of mutating one node in the representation.

    Returns
    -------
    point_mtn : function
        The function which implements the point mutation for GP.
    """
    def point_mtn(repr_):
        """ Implements the point mutation

        The point mutation randomly replaces some randomly selected
        nodes from the individual's representation. The terminals are
        replaced by other terminals and functions are replaced by other
        functions with the same arity.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Creates a copy of parent's representation
        repr_copy = copy.deepcopy(repr_)
        # Performs point replacement
        for i, node in enumerate(repr_copy):
            if random.random() < prob:
                if isinstance(node, _Function):
                    # Finds a valid replacement with same arity
                    node_ = sspace["function_set"][random.randint(0, len(sspace["function_set"])-1)]
                    while node.arity != node_.arity:
                        node_ = sspace["function_set"][random.randint(0, len(sspace["function_set"]) - 1)]
                    # Performs the replacement, once a valid function was found
                    repr_copy[i] = node_
                else:
                    if random.random() < sspace["p_constants"]:
                        repr_copy[i] = sspace["constant_set"][random.randint(0, len(sspace["constant_set"]) - 1)]
                    else:
                        repr_copy[i] = random.randint(0, sspace["n_dims"] - 1)

        return repr_copy

    return point_mtn


def prm_subtree_mtn(initializer):
    """ Implements the the subtree mutation

    This function is used to provide the gs_xo (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator, required to perform the mutation itself.

    Parameters
    ----------
    initializer : function
        Parametrized initialization function to generate random trees.

    Returns
    -------
    subtree_mtn : function
        The function which implements the sub-tree mutation for GP.
    """
    def subtree_mtn(repr_):
        """ Implements the the subtree mutation

        The subtree mutation (a.k.a. standard GP's mutation) replaces a
        randomly selected subtree of the parent individual by a completely
        random tree.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from replacing a randomly selected
            subtree in the parent by a random tree.
        """
        # Generates a random tree
        random_tree = initializer()
        # Calls swap crossover to swap repr_ with random_tree
        return swap_xo(repr_, random_tree)[0]

    return subtree_mtn


def prm_gs_mtn(initializer, ms):
    """ Implements the geometric semantic mutation (GSM)

    This function is used to provide the gs_mtn (inner function) with
    the necessary environment (the outer scope) - the random trees'
    generator and the mutation's step(s). The former is necessary to
    generate a random tree that is required to perform the mutation
    itself, whereas the latter is used to moderate random tree's effect
    on the parent tree.

    Parameters
    ----------
    initializer : float
        Parametrized initialization function to generate random trees.
    ms : torch.Tensor
        A 1D tensor of length m. If it is a single-valued tensor, then
        the mutation step equals ms; if it is a 1D tensor of length m,
        then the mutation step is selected from it at random, at each
        call of gs_mtn.

    Returns
    -------
    gs_mtn : function
        A function which implements the GSM.
    """
    def gs_mtn(repr_):
        """ Implements the geometric semantic mutation (GSM)

        The GSM corresponds to the ball mutation in the semantic space.
        This function stores individuals' representations (the trees)
        in memory.

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        list
            The offspring obtained from adding a random tree, which
            output is bounded in [-ms, ms].
        """
        ms_ = ms if len(ms) == 1 else ms[random.randint(0, len(ms) - 1)]
        return [add2] + repr_ + [mul2, ms_, tanh1] + initializer()

    return gs_mtn


def prm_efficient_gs_mtn(X, initializer, ms):
    """ Implements the an efficient variant of GSM

    This function is used to provide the efficient_gs_mtn (inner
    function) the necessary environment (the outer scope) - the
    input data, the random trees' generator and the mutation's step(s).

    Parameters
    ----------
    X : torch.tensor
        The input data.
    initializer : function
        Initialization function. Used to generate random trees.
    ms : torch.Tensor
        A 1D tensor of length m. If it is a single-valued tensor, then
        the mutation step equals ms; if it is a 1D tensor of length m,
        then the mutation step is selected from it at random, at each
        call of gs_mnt.

    Returns
    -------
    efficient_gs_mtn : function
        A function which implements the efficient GSM.
    """
    def efficient_gs_mtn(repr_):
        """ Implements the an efficient variant of GSM

        Implements an efficient variant of GSM that acts on solutions'
        semantics instead of trees' structure. That is, the trees are
        never stored in computers' memory, only one random tree is
        temporarily generated at each function call to allow the
        calculations to happen (its output and other features are
        stored).
         For more details, consult "A new implementation of geometric
        semantic GP and its application to problems in pharmacokinetics"
        by L. Vanneschi et at. (2013).

        Parameters
        ----------
        repr_ : list
            Parent's representation.

        Returns
        -------
        torch.Tensor
            The offspring's representation stored as semantics vector,
            obtained from adding a random tree bounded in [-ms, ms].
        list
            Random tree generated to perform the GSM.
        torch.Tensor
            The GSM's mutation step used to create the offspring.
        """
        # Chooses the mutation step
        ms_ = ms if len(ms) == 1 else ms[random.randint(0, len(ms) - 1)]
        # Creates a random tree bounded in [-1, 1]
        rt = [tanh1] + initializer()
        # Performs GSM and returns the semantics, the random tree and the mutation's step
        return repr_ + ms_ * _execute_tree(rt, X), rt, ms_

    return efficient_gs_mtn
# ---------------------------


# +++++++++++++++++++++++++++ TSP
def partially_mapped_xo(p1, p2):
    """ Implements the the partially mapped crossover

    After choosing two random cut points on the parents' strings, the
    sub-set between the cut points of one parent's is mapped with the
    other parent's string and the remaining information is exchanged.

    Parameters
    ----------
    p1 : torch.Tensor
        Representation of the first parent.
    p2 : torch.Tensor
        Representation of the second parent.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Resulting offsprings' representations.
    """
    # Generates the cut (ensures both values are different)
    lb, ub = 0, 0
    while lb == ub:
        lb, ub = torch.sort(torch.randint(low=0, high=len(p1), size=(2,)))[0]

    idxs_in = torch.arange(lb, ub+1)
    idxs_out = torch.cat([torch.arange(0, lb), torch.arange(ub+1, len(p1))])
    idxs = torch.arange(0, len(p1))
    cut = torch.logical_and(idxs >= lb, idxs <= ub)

    # Gets the intra-cut representation
    o1 = p1.clone()
    o1[cut] = p2[cut]
    o2 = p2.clone()
    o2[cut] = p1[cut]

    # Defines a recursive function
    def rec(idxs_out_, idxs_in_, out, o1, o2):
        if out == len(idxs_out_):
            pass
        elif o1[idxs_out_[out]] in o1[idxs_in_]:
            o1[idxs_out_[out]] = o2[idxs_in_][(o1[idxs_out_[out]] == o1[idxs_in_]).nonzero()[0]]
            if o1[idxs_out_[out]] in o1[idxs_in_]:
                return rec(idxs_out_, idxs_in_, out, o1, o2)
            else:
                return rec(idxs_out_, idxs_in_, out + 1, o1, o2)
        else:
            return rec(idxs_out_, idxs_in_, out + 1, o1, o2)

    # Applies the recursive function to o1 and o2, respectively
    rec(idxs_out, idxs_in, 0, o1, o2)
    rec(idxs_out, idxs_in, 0, o2, o1)

    return o1, o2


def prm_iswap_mnt(prob):
    """ Implements the the swap mutation

    This function is used to provide the iswap_mnt (inner function)
    the necessary environment (the outer scope) - the probability
    of applying the swap mutation an a given index.

    Parameters
    ----------
    prob : float
        Probability of swapping an index with some other randomly
        selected index.

    Returns
    -------
    iswap_mnt : function
        A function which performs the swap mutation.
    """
    def iswap_mnt(repr_):
        """ Implements the the swap mutation

        Iterates parent's representation and applies the swap mutation,
        index by index, with probability prob.

        Parameters
        ----------
        repr_ : torch.Tensor
            Parent's representation.

        Returns
        -------
        torch.Tensor
            Offspring's representation.
        """
        # Creates a copy of the parent solution
        mutant = repr_.clone()
        for idx in range(len(mutant)):
            if random.uniform(0, 1) < prob:
                rand_idx = random.randint(0, len(repr_)-1)
                rand_val = mutant[rand_idx].item()
                mutant[rand_idx] = mutant[idx]
                mutant[idx] = rand_val
        return mutant

    return iswap_mnt
# ---------------------------


# +++++++++++++++++++++++++++ Knapsack
def one_point_xo(repr_1, repr_2):
    """ Implements the one point crossover

    Selects a random crossover point and swaps parents' representations
    in respect to that point.

    Parameters
    ----------
    repr_1 : Object
        Representation of the first parent.
    repr_2 : Object
        Representation of the second parent.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Resulting offsprings' representations.
    """
    len_ = len(repr_1)
    point = random.randint(0, len_ - 1)
    offsp_repr_1 = torch.cat((repr_1[0:point], repr_2[point:len_]))
    offsp_repr_2 = torch.cat((repr_2[0:point], repr_1[point:len_]))

    return offsp_repr_1, offsp_repr_2


def prm_npoints_xo(n=2):
    """ Implements the N-points crossover

    This function is used to provide the npoints_xo (the inner function)
    the necessary environment (the outer scope) - the number of
    crossover points that will be randomly generated before every
    application of the operator.

    Parameters
    ----------
    n : int (default=2)
        The number of crossover points.

    Returns
    -------
    npoints_xo : function
        A function which performs the N-points crossover.
    """
    def npoints_xo(repr_1, repr_2):
        """ Implements the N-points crossover

        Selects a  n random crossover points and swaps parents'
        representations in respect to those points.

        Parameters
        ----------
        repr_1 : Object
            Representation of the first parent.
        repr_2 : Object
            Representation of the second parent.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Resulting offsprings' representations.
        """
        points = np.random.choice(len(repr_1), size=n, replace=False)
        points = np.sort(points)
        # points = np.array([3, 6, 11])
        print(points)
        offsp_repr_1 = repr_1.clone()
        offsp_repr_2 = repr_2.clone()
        for i in range(len(points)):
            if i % 2 == 0:
                if i == (len(points) - 1):
                    offsp_repr_2[points[i]:] = repr_1[points[i]:]
                    offsp_repr_1[points[i]:] = repr_2[points[i]:]
                else:
                    offsp_repr_2[points[i]:points[i + 1]] = repr_1[points[i]:points[i + 1]]
                    offsp_repr_1[points[i]:points[i + 1]] = repr_2[points[i]:points[i + 1]]

        return offsp_repr_1, offsp_repr_2

    return npoints_xo


def prm_ibinary_flip(prob=0.2):
    """ Implements the iterative binary flip mutation

    This function is used to provide the ibinary_flip (inner function)
    the necessary environment (the outer scope) - the probability
    of applying the binary flip mutation an a given index.

    Parameters
    ----------
    prob : float (default=0.2)
        Probability of inverting a randomly selected (binary) value in
        the solution's representation.

    Returns
    -------
    ibinary_flip : function
        A function which performs the binary flip mutation.
    """
    def ibinary_flip(repr_):
        """ Implements the iterative binary flip mutation

        Iterates parent's representation and applies the binary flip
        mutation, index by index, with probability prob.

        Parameters
        ----------
        repr_ : torch.Tensor
            Parent's representation (stored as floating-points for the
            sake of computational convenience).

        Returns
        -------
        torch.Tensor
            Offspring's representation.
        """
        # Creates a copy of the parent's representation
        mutant = repr_.clone()
        # Creates a random mask with probability prob (where to apply the perturbation)
        rand_mask = torch.bernoulli(torch.tensor([prob] * len(repr_))).bool()
        # Flips the values
        mutant[rand_mask] = torch.where(mutant[rand_mask] == 1.0, 0.0, 1.0)
        # Returns the result
        return mutant

    return ibinary_flip


def prm_rnd_int_ibound(prob=0.5, lb=0, ub=2):
    """ Implements the the discrete ball-like mutation
    
    This function is used to provide the rnd_int_ibound (inner function)
    the necessary environment (the outer scope) - the probability of
    applying the mutation operator at a given index, the lower and the
    upper bounds for random values' generation.

    Parameters
    ----------
    prob : float (default=0.5)
        Probability of applying the operator at a given index.
    lb : int (default=0)
        Lower bound for random values' generation.
    ub : int (default=2)
        Upper bound for random values' generation.

    Returns
    -------
    rnd_int_ibound : function
        A function which performs the discrete ball-like mutation.
    """
    def rnd_int_ibound(repr_):
        """ Implements the the discrete ball-like mutation

        Iterates parent's representation and, with the user-specified
        probability prob, replaces the value at a given index with a
        randomly generated integer ~ U {lb, ub}.

        Parameters
        ----------
        repr_ : torch.Tensor
            Parent's representation.

        Returns
        -------
        torch.Tensor
            Offspring's representation.
        """
        rand_mask = torch.bernoulli(torch.tensor([prob]*len(repr_))).bool()
        n = rand_mask.sum()
        mutant = repr_.clone()
        mutant[rand_mask] = torch.randint(low=lb, high=ub, size=(n,), dtype=torch.float, device=repr_.device)

        return mutant

    return rnd_int_ibound
# ---------------------------


# +++++++++++++++++++++++++++ Continuous Function
def geometric_xo(p1, p2):
    """ Implements the the geometric crossover

    Geometric crossover is a representation-independent generalization
    of the traditional crossover for binary strings. It is defined
    using the distance associated to the solve space in a simple
    geometric way, using the notion of line segment. The main property
    of this operator is the fact it always produces an offspring that
    is never worse than the worst of the parents.
     For more details, consult "Towards a geometric unification of
    evolutionary algorithms" by A. Moraglio (2007).

    Parameters
    ----------
    p1 : torch.Tensor
        First parent's representation.
    p1 : torch.Tensor
        Second parent's representation.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Tuple of two tensors, each representing an offspring obtained
        from applying the geometric crossover.
    """
    # Generates a vector r ~ U(0, 1)
    r = torch.empty(p1.shape[0], device=p1.device).uniform_(0, 1)
    # Creates another vector as 1 - r
    r_ = torch.ones(r.shape[0], device=r.device).sub(r)
    # Performs geometric crossover and returns the result
    return torch.add(torch.mul(r, p1), torch.mul(r_, p2)), torch.add(torch.mul(r, p2), torch.mul(r_, p1))


def prm_iball_mtn(prob, radius):
    """ Implements the iterative ball mutation

    This function is used to provide the iball_mnt (inner function) with
    the necessary environment (the outer scope) - the probability of
    applying the operator at a given index and the mutation's radius.

    Parameters
    ----------
    prob : float
        Probability of applying the ball mutation at a given index of the
        parent's representation.
    radius : float
        Mutation's radius.

    Returns
    -------
    iball_mnt : function
        A function which performs the ball mutation.
    """
    def iball_mnt(repr_):
        """ Implements the iterative ball mutation

        Performs a perturbation of the values at randomly selected
        indexes ~ U(-radius, radius).

        Parameters
        ----------
        repr_ : torch.Tensor
            Parent's representation.

        Returns
        -------
        torch.Tensor
            Offspring's representation.
        """
        # Extrapolates representation's device
        device = repr_.device
        # Creates a copy of the parent solution
        neighbor = repr_.clone()
        # Creates a random mask with probability prob (where to apply the perturbation)
        rand_mask = torch.bernoulli(torch.tensor([prob] * len(repr_))).bool()
        # Performs the random perturbation of the coordinates
        neighbor[rand_mask] += torch.distributions.uniform.Uniform(-radius, radius).sample([rand_mask.sum()]).to(device)
        # Returns the result
        return neighbor

    return iball_mnt


def de_binomial_xo(donor, target, c_rate):
    """ Implements the binomial crossover for DE

    Creates the trial vector from the elements of the donor vector
    (with probability c_rate) and the target vector (with 1-c_rate).

    Parameters
    ----------
    donor : torch.Tensor
        The donor vector at a given iteration.
    target : torch.Tensor
        The target vector at a given iteration.
    c_rate : float
        The probability of selecting elements from the donor vector
        (a.k.a. crossover's rate).

    Returns
    -------
    torch.Tensor
        The trial vector consisting of the elements from the donor
        and the target vectors.
    """
    mask = torch.rand(len(target), device=target.device) <= c_rate
    return torch.where(condition=mask, input=donor, other=target)


def de_exponential_xo(donor, target, c_rate):
    """ Implements the exponential crossover for DE

    Creates the trial vector from the elements of the donor vector
    and target vectors. The values between the index r and r + l,
    where r ~ U{0, len(target)-1} and l ~ U{0, len(target)-1-r},
    are taken from the donor vector until the Bernoulli experiment
    with success probability c_rate will fail for the first time or
    all the aforementioned indexes of the donor were exchanged. All
    the remaining parameters are taken from the target vector.

    Parameters
    ----------
    donor : torch.Tensor
        The donor vector at a given iteration.
    target : torch.Tensor
        The target vector at a given iteration.
    c_rate : float
        The probability of selecting elements from the donor vector
        (a.k.a. crossover's rate).

    Returns
    -------
    trial : torch.Tensor
        The trial vector consisting of the elements from the donor
        and the target vectors.
    """
    # Sets the trial as the copy of the target
    trial = target.clone()
    # Generates the starting point for the exchange
    r = random.randint(0, len(donor) - 1)
    # Generates the number of components the donor will contribute
    l = random.randint(0, len(donor) - 1 - r)
    # Loop
    while random.random() <= c_rate and l > 0:
        # Exchange
        trial[r] = donor[r]
        # Switch index
        r += 1
        # Update exchanged components' count
        l -= 1

    return trial


def de_rand(parents, m_weights):
    """ Implements the DE/RAND/N mutation

    Implements the DE/RAND/N mutation's strategy following the
    formula 𝑉(𝐺+1) = 𝑉(r1, 𝐺) + 𝐹1[𝑉(r2, 𝐺) − V(r3, 𝐺)] + (...) +
    𝐹N[𝑉(r(N*2), 𝐺) − V(r(N*2+1), 𝐺)]. Note that 'RAND' stands for the
    random selection of the base vector while N for the number of
    differences' pairs of randomly selected parents.

    Parameters
    ----------
    parents : torch.Tensor
        A tensor consisting of n randomly selected parents. The number
        of parents should be equal to N*2+1.
    m_weights : torch.Tensor
        A tensor containing the N mutation's weights (one per random
        difference).

    Returns
    -------
    donor : torch.Tensor
        The donor vector (i.e., the mutant).
    """
    donor, j = parents[0].clone(), 1
    for w in m_weights:
        donor += w * torch.sub(parents[j], parents[j + 1])
        j += 2

    return donor


def de_best(best, parents, m_weights):
    """ Implements the DE/BEST/N mutation

    Implements the DE/BEST/N mutation's strategy following the
    formula 𝑉(𝐺+1) = 𝑉(𝑏𝑒𝑠𝑡,𝐺) + 𝐹1[𝑉(r2, 𝐺) − V(r3, 𝐺)] + (...) +
    𝐹N[𝑉(r(N*2-1), 𝐺) − V(r(N*2), 𝐺)]. Note that 'BEST' stands for the
    random selection of the base vector while N for the number of
    differences' pair of randomly selected parents.

    Parameters
    ----------
    best : torch.Tensor
        A tensor representing the best parent in the population.
    parents : torch.Tensor
        A tensor consisting of n randomly selected parents. The number
        of parents should be equal to N*2.
    m_weights : torch.Tensor
        A tensor containing the N mutation's weights (one per random
        difference).

    Returns
    -------
    donor : torch.Tensor
        The donor vector (i.e., the mutant)
    """
    donor, j = best.clone(), 0
    for w in m_weights:
        donor += w * torch.sub(parents[j], parents[j + 1])
        j += 2

    return donor


def de_target_to_best(target, best, parents, m_weights):
    """ Implements the DE/TARGET-TO-BEST/N mutation

    Implements the DE/RAND-TO-BEST/N mutation's strategy following the
    formula 𝑉(𝐺+1) = 𝑉(𝑖,𝐺) + F1[𝑉(𝑏𝑒𝑠𝑡,𝐺) − 𝑉(𝑖,𝐺)] + 𝐹2[𝑉(r1,𝐺) −
    𝑉(r2,𝐺)] + (...) + 𝐹N[𝑉(r(N*2-1), 𝐺) − V(r(N*2), 𝐺)]. Note that
    'TARGET' stands for the target vector's selection as the base
    vector while N for the number of differences' pair of randomly
    selected parents. Moreover, this mutation strategy includes an
    additional weighted differences pair between the elite and the
    underlying base vector.

    Parameters
    ----------
    target : torch.Tensor
        The target vector i of iteration G - 𝑉(𝑖,𝐺).
    best : torch.Tensor
        A tensor representing the best parent in the population.
    parents : torch.Tensor
        A tensor consisting of n randomly selected parents. The number
        of parents should be equal to N*2.
    m_weights : torch.Tensor
        A tensor containing the N mutation's weights (one per random
        difference).

    Returns
    -------
    donor : torch.Tensor
        The donor vector (i.e., the mutant)
    """
    donor, j = target + m_weights[0]*torch.sub(best, target), 0
    for wi in range(len(m_weights)-1):
        donor += m_weights[wi] * torch.sub(parents[j], parents[j + 1])
        j += 2

    return donor


def prm_pso(c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
    """ PSO's update rule.

    Provides pso (the inner function) with the necessary outer scope
    consisting of the social and cognitive factors, and the update's
    maximum and minimum inertia. The latter will decrease over the time
    in a linear fashion.

    Parameters
    ----------
    c1 : float (default=2.0)
        the influence of social factor (𝑪𝟏).
    c2 : float (default=2.0)
        the influence of cognitive factor (𝑪𝟐).
    w_max : float (default=0.9)
        Initial inertia parameter (at the first iteration).
    w_min : float (default=0.4)
        Final inertia parameter (at the last iteration).

    Returns
    -------
    pso : function
        A function which performs particles' update given the social
        and cognitive factors, and the inertia parameters.
    """
    # Performs the minimum necessary checks
    update_inertia = True
    if w_max == w_min:
        update_inertia = False
    elif w_min is None:
        update_inertia = False

    def pso(pos_p, v_p, lbest_p, gbest, it, it_max):
        """ PSO's update rule.

        Parameters
        ----------
        pos_p : torch.Tensor
            Particle's current position.
        v_p : torch.Tensor
            Particle's previous velocity vector.
        lbest_p : torch.Tensor
            Particle's local best position.
        gbest : torch.Tensor
            Swarm's global best position.
        it : int
            Current iteration (used for inertia's update).
        it_max : int
            Maximum number of iterations (used for inertia's update).

        Returns
        -------
        torch.Tensor
            The new velocity vector (𝒗_𝒑[𝒕]) after PSO's update rule.
        """
        # Computes current inertia weight
        w_i = w_max - ((w_max - w_min) / it_max) * it if update_inertia else w_max
        # Generates the random constants
        r1 = torch.rand(pos_p.shape, device=pos_p.device)
        r2 = torch.rand(pos_p.shape, device=pos_p.device)
        # Computes new velocity vector
        v_p_new = w_i * v_p + c1 * r1 * (gbest - pos_p) + c2 * r2 * (lbest_p - pos_p)

        return v_p_new

    return pso
# ---------------------------
