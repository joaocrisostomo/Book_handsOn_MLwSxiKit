import torch


class Population:
    """ Implementation of a Solution class for any OP.

   The purpose of a Search Algorithm (SA) is to solve a given
   Optimization Problem (OP). The solve process consists of travelling
   across the solve space (𝑆) in a specific manner (which is embedded
   in algorithm's definition). Some algorithms manipulate the whole
   set of solutions at a time to perform such a solve. For this
   reason, in scope of this library, a special class to efficiently
   encapsulate the whole population of candidate solutions was created.
   More specifically, to avoid redundant generation of objects to store
   a set of solution, their essential characteristics will be
   efficiently stored as a limited set of macro-objects, all
   encapsulated in class Population.


    Attributes
    ----------
    _pop_id : int
        A unique identification of a population object.
    repr_ : Object
        The solutions' representation in the population.
    valid : torch.Tensor
        The solutions' validity state under the light of 𝑆.
    fit : torch.Tensor
        A tensor representing solutions' quality in 𝑆. It is assigned
        by a given problem instance (PI), using fitness function (𝑓).
    """
    pop_id = 0

    def __init__(self, repr_):
        """ Object's constructor.

        Parameters
        ----------
        repr_ : Object
            The solutions' representation in the population.
        """
        self._pop_id = Population.pop_id
        Population.pop_id += 1
        self.repr_ = repr_
        self.valid = None
        self.fit = None

    def _get_copy(self):
        """ Makes a copy of the calling Population object.

        Returns
        -------
        pop : Population
            An object of type Population, copy of self.
        """
        if type(self.repr_) is torch.Tensor:
            pop_copy = Population(self.repr_.clone())
        else:
            pop_copy = Population(self.repr_.copy())
        if hasattr(self, 'valid'):
            pop_copy.valid = self.valid
        if hasattr(self, 'fit'):
            pop_copy.fit = self.fit.clone()
        if hasattr(self, 'test_fit'):
            pop_copy.val_fit = self.val_fit.clone()

        return pop_copy

    def __len__(self):
        return len(self.repr_)

    def __getitem__(self, index):
        return self.repr_[index]
