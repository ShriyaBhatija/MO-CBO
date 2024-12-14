import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from .mocbo3_CostFunctions import define_costs


class MO_CBO3(GraphStructure):
    
    def __init__(self, observational_samples):

        self.U = np.asarray(observational_samples['U'])[:,np.newaxis]
        self.X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        self.X3 = np.asarray(observational_samples['X3'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]
        self.Y3 = np.asarray(observational_samples['Y3'])[:,np.newaxis]
        self.control_node = np.asarray(observational_samples['control'])[:,np.newaxis]

    def define_SEM(self):
        
        def fU(epsilon, **kwargs):
          return np.random.choice([-5, 5], p=[0.5, 0.5])
        
        def fx1(epsilon, **kwargs):
          return epsilon[0]
        
        def fx2(epsilon, **kwargs):
          return epsilon[1]
          
        def fx3(epsilon, U, **kwargs):
          return U + epsilon[2]

        def fy1(epsilon, X1, X2, **kwargs):
          return X1**2 + X2**2 + epsilon[3]

        def fy2(epsilon, X1, X2, **kwargs):
          return (X2-5)**2 + (X1-10)**2 + epsilon[4]
        
        def fy3(epsilon, X2, X3, U, **kwargs):
          return (X2-10)**2 -  X3*U + epsilon[5]
        
        # This is just a buffer, so the code works for intervention sets of length one
        # Note that this does not influence any other variables (i.e. it is parent and childless)
        def f_control(epsilon, **kwargs):
          return epsilon[7]*0

        graph = OrderedDict ([
          ('U', fU),
          ('X1', fx1),
          ('X2', fx2),
          ('X3', fx3),
          ('Y1', fy1),
          ('Y2', fy2),
          ('Y3', fy3),
          ('control', f_control)
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2', 'Y3'] 
    

    def get_exploration_sets(self):
      POMIS = [['X1','X2'], ['X1', 'X2', 'X3']]
      manipulative_variables = [['X1', 'X2', 'X3']]

      exploration_sets = {
         'pomis': POMIS,
         'mobo': manipulative_variables
      }
      return exploration_sets
    

    def get_set_MOBO(self):
      manipulative_variables = ['X1', 'X2', 'X3']
      return manipulative_variables
    

    def get_interventional_ranges(self):

      dict_ranges = OrderedDict ([
          ('X1', [0, 5]),
          ('X2', [0, 5]),
          ('X3', [-5, 5]),
          ('control', [-0.01, 0.01])
        ])
      
      return dict_ranges
    
    def fit_all_models(self):
       pass

    def refit_models(self, observational_samples):
       pass

    def get_all_do(self):
       pass
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs