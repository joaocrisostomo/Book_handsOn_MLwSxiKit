import math
import torch


# TODO: 1) check what to do with i_vec; 2) force x to always have a shape of [n, D], even if D=1
def sphere_function(x):
    """ Implements Spherical function (De Jong’s)

    Uni-modal function.

    Parameters
    ----------
    x : torch.Tensor
        ...

    Returns
    -------
    torch.Tensor
        ...
    """
    n_dims = (len(x.shape)-1)
    return torch.sum(torch.pow(x, 2.0), dim=n_dims)


def rastrigin_function(x):
    """ Implements Spherical function (De Jong’s)

    Multi-modal function.
    """
    n_dims = (len(x.shape)-1)
    return 10.0 * x.shape[n_dims] + torch.sum(x ** 2 - 10.0 * torch.cos(math.pi * 2.0 * x), dim=n_dims)


def ackley_function(x):
    """ Implements Ackley function.

    Multi-modal function.
    """
    n_dims = (len(x.shape)-1)
    return 20.0 + torch.exp(torch.tensor([1.0])).item() - 20.0 * \
            torch.exp(- 0.2 * torch.sqrt(torch.mean(x**2, dim=n_dims))) - \
            torch.exp(torch.mean(torch.cos(math.pi * 2.0 * x), dim=n_dims))


def rosenbrock_function(x):
    """ Implements Rosenbrock function.

    Uni-modal function.
    """
    n_dims = (len(x.shape)-1)
    if n_dims == 0:
        return torch.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2, dim=n_dims)
    else:
        return torch.sum((1 - x[:, :-1]) ** 2 + 100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2, dim=n_dims)


def param_griewank_function(i_vec):
    """
    n_dims = (len(x.shape) - 1)
    x_dim = len(x) if n_dims == 0 else len(x[0])
    return (1.0 / 4000.0) * torch.sum(x ** 2, dim=n_dims) - \
           torch.prod(torch.cos(torch.div(x, torch.sqrt(torch.arange(1.0, (x_dim + 1.0), device=x.device)))),
                      dim=n_dims) + 1.0
    """
    def griewank_function(x):
        """ Implements Griewank function.

        Multi-modal function.
        """
        n_dims = (len(x.shape)-1)
        return (1.0/4000.0) * torch.sum(x**2, dim=n_dims) - torch.prod(torch.cos(torch.div(x, torch.sqrt(i_vec))), dim=n_dims) + 1.0

    return griewank_function


def salomon_function(x):
    """ Implements Salomon function.

    Multi-modal function.
    """
    n_dims = (len(x.shape) - 1)
    return 1.0 - torch.cos(2 * math.pi * torch.sum(x ** 2, dim=n_dims)) + 0.1 * torch.sum(x ** 2, dim=n_dims)


def param_quartic_function(i_vec):
    """

    """
    def quartic_function(x):
        """ Implements Quartic function.

        Uni-modal function.
        """
        n_dims = len(x.shape) - 1
        return torch.sum(torch.pow(x, 4.0) * i_vec, dim=n_dims)

    return quartic_function


def param_hyperellipsoid_function(i_vec):
    """ Implements Hyper-ellipsoid function.

    Uni-modal function.

    Parameters
    ----------
    i_vec : torch.Tensor

    Returns
    -------
    hyperellipsoid_function : function
        Parametrized Hyper-ellipsoid function.
    """
    def hyperellipsoid_function(x):
        """ Parametrized Hyper-ellipsoid function.

        Parameters
        ----------
        x : torch.Tensor
            The input data of shape [D] or [n, D], where n represents
            the number of data instances and D the number of dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of shape [n].

        """
        n_dims = len(x.shape) - 1
        return torch.sum(torch.pow(x, 2.0) * i_vec, dim=n_dims)

    return hyperellipsoid_function


def mexican_hat_function(x):
    """ Implements the Mexican hat wavelet (2D).

    The Mexican hat wavelet function is defined exclusively in 1 and 2
    dimensions. This function implements the wavelet in 2D. The
    hypercube where it is usually evaluated is defined as 𝑥_𝑖 ∈
    [−5.0, 5.0], where 𝑖 represents one of the dimensions.

    Parameters
    ----------
    x : torch.Tensor
        The input data of shape [2] or [n, 2], where n represents
        the number of data instances in the 2D hypercube.

    Returns
    -------
    torch.Tensor
        Tensor of shape [n].
    """
    n_dims = (len(x.shape)-1)
    if n_dims == 0:
        return (1/math.pi)*(1.0 - 0.5*(x[0]**2 - x[1]**2))*torch.exp(-(x[0]**2 + x[1]**2)/2.0)
    else:
        return (1/math.pi)*(1.0 - 0.5*(x[:, 0]**2 - x[:, 1]**2))*torch.exp(-(x[:, 0]**2 + x[:, 1]**2)/2.0)


def param_branin_function(a=1.0, b=5.1 / (4 * (math.pi ** 2)), c=5 / math.pi, r=6, s=10, t=1 / (8 * math.pi)):
    """ Implements the Branin function.

    The Branin function is defined exclusively in 2D and has three
    global minima; usually it is evaluated on the square 𝑥_1 ∈ [−5, 10],
    𝑥_2 ∈ [0, 15].


    Parameters
    ----------
    a : float
    b : float
    c : float
    r : float
    s : float
    t : float

    Returns
    -------
    branin_function : function
        The Branin function parametrized with a, b, c, r, s and t.
    """
    def branin_function(x):
        """ Parametrized 2D Branin function.

        Parameters
        ----------
        x : torch.Tensor
            The input data of shape [2] or [n, 2], where n represents
            the number of data instances in the 2D hypercube.

        Returns
        -------
        torch.Tensor
            Tensor of shape [n].
        """
        n_dims = (len(x.shape) - 1)
        if n_dims == 0:
            return a * (x[1] - b * (x[0] ** 2) + c * x[0] - r) ** 2 + s * (1 - t) * torch.cos(x[0]) + s
        else:
            return a * (x[:, 1] - b * (x[:, 0] ** 2) + c * x[:, 0] - r) ** 2 + s * (1 - t) * torch.cos(x[:, 0]) + s

    return branin_function


def discus_function(x):
    """ Implements the Discus function.

    The Discus function is a quadratic function which has local
    irregularities. A single direction in solve space is a thousand
    times more sensitive than all others. The function is defined in
    𝑁  dimensions. Researchers sugges a regular hypercube defined as
    𝑥_𝑖 ∈ [−32.786, 32.786].

    Parameters
    ----------
    x : torch.Tensor
        The input data of shape [D] or [n, D], where n represents
        the number of data instances and D the number of dimensions.

    Returns
    -------
    torch.Tensor
        Tensor of shape [n].
    """
    n_dims = (len(x.shape)-1)
    if n_dims == 0:
        return (10**6)*(x[0]**2) + torch.sum(x[1:]**2)
    else:
        return (10**6)*(x[:, 0]**2) + torch.sum(x[:, 1:]**2, n_dims)


def kotanchek_function(x):
    """ Implements the Kotanchek function.

    The Kotanchek function is defined exclusively in 2D. It is
    usually evaluated is defined in 𝑥_1 ∈ [−2, 7], 𝑥_2 ∈ [−1, 3].

    Parameters
    ----------
    x : torch.Tensor
        The input data of shape [2] or [n, 2], where n represents
        the number of data instances in the 2D hypercube.

    Returns
    -------
    torch.Tensor
        Tensor of shape [n].
    """
    n_dims = (len(x.shape) - 1)
    if n_dims == 0:
        return torch.exp(-(x[0] - 1.0) ** 2) / (1.2 + (x[1] - 2.5) ** 2)
    else:
        return torch.exp(-(x[:, 0] - 1.0) ** 2) / (1.2 + (x[:, 1] - 2.5) ** 2)


def param_weierstrass_function(a=0.5, b=3, kmax=20):
    """ Implements the Weierstrass function.

    The function is usually evaluated in a regular hypercube defined as
    𝑥_𝑖 ∈ [−0.5, 0.5], where 𝑖 represents a given dimension.


    Parameters
    ----------
    a : float
    b : float
    kmax : float

    Returns
    -------
    weierstrass_function : function
        Parametrized Weierstrass function.
    """
    def weierstrass_function(x):
        """ Parametrized Weierstrass function.

        Parameters
        ----------
        x : torch.Tensor
            The input data of shape [D] or [n, D], where n represents
            the number of data instances and D the number of dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of shape [n].
        """
        D = x.shape[1] if len(x.shape) == 2 else 1
        k_ = torch.arange(0, kmax).float().to(x.device)
        a_ = torch.pow(a, k_)[:, None]
        b_ = torch.pow(b, k_)[:, None]

        D_a_cos_pi_b_ = D * torch.matmul(a_.T, torch.cos(math.pi * b_))[0]

        if D == 1:
            return torch.matmul(a_.T, torch.cos(torch.matmul(math.pi * b_, x[None, :])))[0]
        else:
            a_cos_2pi_b_xi05 = None

            for d_i in range(D):
                xi = x[:, d_i][None, :]
                if d_i == 0:
                    a_cos_2pi_b_xi05 = torch.matmul(a_.T,
                                                    torch.cos(torch.matmul(2 * math.pi * b_, xi + 0.5))) - D_a_cos_pi_b_
                else:
                    a_cos_2pi_b_xi05 += torch.matmul(a_.T, torch.cos(
                        torch.matmul(2 * math.pi * b_, xi + 0.5))) - D_a_cos_pi_b_

            return torch.sum(a_cos_2pi_b_xi05, 0)

    return weierstrass_function