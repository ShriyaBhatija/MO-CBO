"""
This code was originally published by the following individual:
    Copyright (c) 2020 Yunsheng Tian
    GitHub: https://github.com/yunshengtian/DGEMO?tab=MIT-1-ov-file
"""


from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from .utils import safe_divide, expand

'''
Acquisition functions that define the objectives for surrogate multi-objective problem
'''

class Acquisition(ABC):
    '''
    Base class of acquisition function
    '''
    requires_std = False # whether requires std output from surrogate model, set False to avoid unnecessary computation

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, Y):
        '''
        Fit the parameters of acquisition function from data
        '''
        pass

    @abstractmethod
    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        '''
        Evaluate the output from surrogate model using acquisition function
        Input:
            val: output from surrogate model, storing mean and std of prediction, and their derivatives
            val['F']: mean, shape (N, n_obj)
            val['dF']: gradient of mean, shape (N, n_obj, n_var)
            val['hF']: hessian of mean, shape (N, n_obj, n_var, n_var)
            val['S']: std, shape (N, n_obj)
            val['dS']: gradient of std, shape (N, n_obj, n_var)
            val['hS']: hessian of std, shape (N, n_obj, n_var, n_var)
        Output:
            F: acquisition value, shape (N, n_obj)
            dF: gradient of F, shape (N, n_obj, n_var)
            hF: hessian of F, shape (N, n_obj, n_var, n_var)
        '''
        pass


class IdentityFunc(Acquisition):
    '''
    Identity function
    '''
    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        F, dF, hF = val['F'], val['dF'], val['hF']
        return F, dF, hF