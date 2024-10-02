import math
from joblib import cpu_count
from scipy.stats import spearmanr, pearsonr

import torch


# +++++++++++++++++++++++++++ Train/Test split
def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """ Splits X and y tensors into train and test subsets

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : torch.Tensor
        Input data instances,
    y : torch.Tensor
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether or not to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether or not to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random numbers generators.

    Returns
    -------
    X_train : torch.Tensor
        Training data instances.
    y_train : torch.Tensor
        Training target vector.
    X_test : torch.Tensor
        Test data instances.
    y_test : torch.Tensor
        Test target vector.
    train_indices : torch.Tensor
        Indices representing the training partition.
    test_indices : torch.Tensor
    Indices representing the test partition.
    """
    # Sets the seed before generating partition's indexes
    torch.manual_seed(seed)
    # Generates random indices
    if shuffle:
        indices = torch.randperm(X.shape[0])
    else:
        indices = torch.arange(0, X.shape[0], 1)
    # Splits indices
    split = int(math.floor(p_test * X.shape[0]))
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        # Generates train/test partitions
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


def _get_tasks_per_job(total, n_jobs):
    """

    Parameters
    ----------


    Returns
    -------
    tasks_per_job : torch.Tensor
    """
    # Verifies parameter's validity
    if n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    # Estimates the effective number of jobs
    n_jobs_ = min(cpu_count(), n_jobs)
    tasks_per_job = (total // n_jobs_) * torch.ones(n_jobs, dtype=torch.int)
    return tasks_per_job[:tasks_per_job % n_jobs] + 1


# +++++++++++++++++++++++++++ Fitness (a.k.a. Cost) Functions
def travel_distance(dist_mtrx, origin, repr_):
    """ Computes the travelling distance for an instance of TSP


    """
    fitness = 0.0
    if len(repr_.shape) == 1:
        fitness += dist_mtrx[origin, repr_[0]]
        fitness += dist_mtrx[repr_[0:-1], repr_[1:]].sum()
        return fitness + dist_mtrx[repr_[-1], origin]
    else:
        fitness += dist_mtrx[origin, repr_[:, 0]]
        fitness += dist_mtrx[repr_[:, 0:-1], repr_[:, 1:]].sum(1)
        return fitness + dist_mtrx[repr_[:, -1], origin]


def prm_asigmoid_rmse(a=0.5):
    """ Implements RMSE for binary classification

    Computes the RMSE after passing the soft prediction through the
    logistic function with  growth rate parameters equal to a.

    References
    ----------
    [1] I. Bakurov, M. Castelli, F. Fontanella, L. Vanneschi, "A
    Regression-like Classification System for Geometric Semantic
    Genetic Programming", 11th International Conference on Evolutionary
    Computation Theory and Applications.

    Parameters
    ----------
    a : float
        Parameter alpha that controls the logistic function's flatness
        (a.k.a. the growth rate).
    """
    def asigmoid_rmse(y_true, y_pred):
        return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, torch.sigmoid(a*y_pred)), 2), len(y_pred.shape)-1))

    return asigmoid_rmse


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1))


def mse(y_true, y_pred):
    return torch.mean(torch.pow(torch.sub(y_true, y_pred), 2), len(y_pred.shape)-1)


def mae(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)))


def mae_int(y_true, y_pred):
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))))


def n_equal(y_true, y_pred):
    return torch.eq(y_true, torch.argmax(y_pred.squeeze(), dim=1).type(y_true.dtype)).sum()


def spearman_cc(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def spearman_cc_protected(y_true, y_pred):
    if y_true.is_cuda:
        y_true_, y_pred_ = y_true.cpu(), y_pred.cpu()
    else:
        y_true_, y_pred_ = y_true, y_pred
    if len(y_true_.shape) < len(y_pred_.shape):
        # When the y_pred_ is computed from the population
        res = torch.tensor([spearmanr(y_true_, y)[0] for y in y_pred_], device=y_true.device)
    elif y_true_.shape > y_pred_.shape:
        # When the tree returns a constant
        res = torch.tensor(spearmanr(y_true_, torch.repeat_interleave(y_pred_, len(y_true_)))[0], device=y_true.device)
    else:
        # When the y_pred_ is computed from a single solution
        res = torch.tensor(spearmanr(y_true_, y_pred_)[0], device=y_true.device)

    # Replaces NaNs
    res[res != res] = 0.0
    return torch.abs(res)


def pearson_cc(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def prm_miou(n_classes):
    """ Computes the mean intersection over union (MIoU)

    This function is used to provide the miou (the inner function)
    with the necessary environment (the outer scope) - the total
    number of target classes in the images.

    Parameters
    ----------
    n_classes : int
        The total number of target classes in the images.

    Returns
    -------
    miou : function
        Function that computes the MiOU given the user-specified
        number of classes.
    """
    def miou(y_true, y_pred):
        """ Computes the mean intersection over union (MIoU)

        Computes the MiOU given the user-specified number of classes.

        Parameters
        ----------
        y_true : torch.Tensor
            The ground truth (correct) labels.
        y_pred : torch.Tensor
            The predicted labels, as returned by a semantic
            segmentation system.

        Returns
        -------
        miou : torch.Tensor
            Mean intersection over union.
        """
        # Takes the maximum index for the logits volume (the segmentation's map)
        y_pred = torch.argmax(y_pred.squeeze(), dim=1)
        y_true = y_true.type(y_pred.dtype)
        # Selects not "I don't care" pixels from the target
        k = (y_true >= 0) & (y_true < n_classes)
        hist = torch.bincount(n_classes * y_true[k] + y_pred[k], minlength=n_classes ** 2)
        hist = hist.reshape(n_classes, n_classes).type(torch.float64)
        iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        miou = torch.mean(iou[iou == iou])
        return miou

    return miou
