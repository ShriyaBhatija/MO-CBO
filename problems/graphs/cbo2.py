import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from utils_functions.utils import fit_single_GP_model

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .cbo2_DoFunctions import *

'''
class CBO2(GraphStructure):
    
    def __init__(self, observational_samples):
                
        self.A = np.asarray(observational_samples['A'])[:,np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]


    def define_SEM(self):

        def fD(epsilon, **kwargs):
          return epsilon[0]
        
        def fC(epsilon, D, **kwargs):
          return epsilon[1]
        
        def fA(epsilon, C, **kwargs):
          return epsilon[2]
        
        def fB(epsilon, A, D, **kwargs):
          return epsilon[3]

        def fY1(epsilon, A, B, **kwargs):
          return epsilon[4]

        def fY2(epsilon, A, B, **kwargs):
          return epsilon[5]

        graph = OrderedDict ([
          ('A', fA),
          ('B', fB),
          ('C', fC),
          ('D', fD),
          ('Y1', fY1),
          ('Y2', fY2),
        ])

        return graph
'''