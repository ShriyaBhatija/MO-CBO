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
                
        self.X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        self.X3 = np.asarray(observational_samples['X3'])[:,np.newaxis]
        self.X4 = np.asarray(observational_samples['X4'])[:,np.newaxis]
        self.X5 = np.asarray(observational_samples['X5'])[:,np.newaxis]
        self.X6 = np.asarray(observational_samples['X6'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]
        self.control_node = np.asarray(observational_samples['control'])[:,np.newaxis]

    def define_SEM(self):

        def fU(epsilon, **kwargs):
          return -4

        def fx4(epsilon, U, **kwargs):
          return U

        def fx5(epsilon, **kwargs):
          return epsilon[1]

        def fx7(epsilon, **kwargs):
          return epsilon[2]

        def fx8(epsilon, **kwargs):
          return epsilon[3]

        def fx6(epsilon, X5, X7, X8, **kwargs):
          return np.exp(X5+X7+X8-30) + epsilon[4]

        def fx1(epsilon, X4, **kwargs):
          return X4/2
        
        def fx2(epsilon, X5, X6, **kwargs):
          return np.exp(X5+X6-10) + (0.1*epsilon[6])**3
        
        def fx3(epsilon, X5, X7, **kwargs):
          return np.log(1+(X5+X7)/10) + (0.1*epsilon[7])**2

        def fy1(epsilon, X1, X2, U, **kwargs):
          return np.log(1+X1**2) + 2*X2**2 - X1*X2*(U/2) 

        def fy2(epsilon, X2, X3, **kwargs):
          return np.sin(X2**2) - X3**2 - X2*X3 + 50 
        
        # This is just a buffer, so the code works for intervention sets of length one
        # Note that this does not influence any other variables (i.e. it is parent and childless)
        def f_control(epsilon, **kwargs):
          return epsilon[10]*0

        graph = OrderedDict ([
          ('U', fU),
          ('X4', fx4),
          ('X5', fx5),
          ('X7', fx7),
          ('X8', fx8),
          ('X6', fx6),
          ('X1', fx1),
          ('X2', fx2),
          ('X3', fx3),
          ('Y1', fy1),
          ('Y2', fy2),
          ('control', f_control)
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_exploration_sets(self):
        MIS = [
                ['X1', 'control'], ['X2', 'control'], ['X3', 'control'], ['X4', 'control'], ['X5', 'control'], ['X6', 'control'], ['X7', 'control'], ['X8', 'control'],
                ['X1', 'X2'], ['X1', 'X3'], ['X1', 'X5'], ['X1', 'X6'], ['X1', 'X7'], ['X1', 'X8'],
                ['X2', 'X4'], ['X2', 'X3'], ['X2', 'X5'], ['X2', 'X7'], ['X2', 'X8'],
                ['X3', 'X4'], ['X3', 'X5'], ['X3', 'X6'], ['X3', 'X7'], ['X3', 'X8'],
                ['X4', 'X5'], ['X4', 'X6'], ['X4', 'X7'], ['X4', 'X8'], 
                ['X5', 'X6'], ['X5', 'X7'], ['X5', 'X8'],
                ['X6', 'X7'],
                ['X7', 'X8']
        ]
        POMIS = [['X2', 'X3']]
        manipulative_variables = [['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]

        exploration_sets = {
            'pomis': POMIS,
            'mobo': manipulative_variables
        }
        return exploration_sets
    

    def get_set_MOBO(self):
        manipulative_variables = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        return manipulative_variables
    

    def get_interventional_ranges(self):
        min_intervention_x1 = 0
        max_intervention_x1 = 5

        min_intervention_x2 = 0
        max_intervention_x2 = 5

        min_intervention_x3 = 0
        max_intervention_x3 = 5

        min_intervention_x_rest = 0
        max_intervention_x_rest = 5

        min_control = -0.5
        max_control = 0.5

        dict_ranges = OrderedDict ([
            ('X1', [min_intervention_x1, max_intervention_x1]),
            ('X2', [min_intervention_x2, max_intervention_x2]),
            ('X3', [min_intervention_x3, max_intervention_x3]),
            ('X4', [min_intervention_x_rest, max_intervention_x_rest]),
            ('X5', [min_intervention_x_rest, max_intervention_x_rest]),
            ('X6', [min_intervention_x_rest, max_intervention_x_rest]),
            ('X7', [min_intervention_x_rest, max_intervention_x_rest]),
            ('X8', [min_intervention_x_rest, max_intervention_x_rest]),
            ('control', [min_control, max_control])
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