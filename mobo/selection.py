from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.nsga2 import calc_crowding_distance

'''
Selection methods for new batch of samples to evaluate on real problem
'''

class Selection(ABC):
    '''
    Base class of selection method
    '''
    def __init__(self, batch_size, ref_point=None, **kwargs):
        self.batch_size = batch_size
        self.ref_point = ref_point

    def fit(self, X, Y):
        '''
        Fit the parameters of selection method from data
        '''
        pass

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    @abstractmethod
    def select(self, solution, surrogate_model, status, transformation):
        '''
        Select new samples from solution obtained by solver
        Input:
            solution['x']: design variables of solution
            solution['y']: acquisition values of solution
            solution['algo']: solver algorithm, having some relevant information from optimization
            surrogate_model: fitted surrogate model
            status['pset']: current pareto set found
            status['pfront]: current pareto front found
            status['hv']: current hypervolume
            transformation: data normalization for surrogate model fitting
            (some inputs may not be necessary for some selection criterion)
        Output:
            X_next: next batch of samples selected
            info: other informations need to be stored or exported, None if not necessary
        '''
        pass


class DGEMOSelect(Selection):
    '''
    Selection method for DGEMO algorithm
    '''
    has_family = True

    def select(self, solution, surrogate_model, status, transformation):
        algo = solution['algo']

        X_next, _, family_lbls_next = algo.propose_next_batch(status['pfront'], self.ref_point, self.batch_size, transformation)
        family_lbls, approx_pset, approx_pfront = algo.get_sparse_front(transformation)

        info = {
            'family_lbls_next': family_lbls_next,
            'family_lbls': family_lbls,
            'approx_pset': approx_pset,
            'approx_pfront': approx_pfront,
        }
        return X_next, info
