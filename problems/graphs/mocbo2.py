import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs



class MO_CBO2(GraphStructure):
    
    def __init__(self, observational_samples):
                
        self.X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        self.X3 = np.asarray(observational_samples['X3'])[:,np.newaxis]
        self.X4 = np.asarray(observational_samples['X4'])[:,np.newaxis]
        self.X5 = np.asarray(observational_samples['X5'])[:,np.newaxis]
        self.X6 = np.asarray(observational_samples['X6'])[:,np.newaxis]
        self.X7 = np.asarray(observational_samples['X7'])[:,np.newaxis]
        self.X8 = np.asarray(observational_samples['X8'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]

    def define_SEM(self):

        def fU(epsilon, **kwargs):
          #return np.random.normal(-4, 0.1, 1)[0]
          return np.random.choice([-4, 4], p=[0.5, 0.5])

        def fx4(epsilon, U, **kwargs):
          return (epsilon[0]/20)**2 + U

        def fx5(epsilon, **kwargs):
          return epsilon[1]

        def fx7(epsilon, **kwargs):
          return epsilon[2]

        def fx8(epsilon, **kwargs):
          return epsilon[3]

        def fx6(epsilon, X5, X7, X8, **kwargs):
          return np.exp(X5+X7+X8-30) + epsilon[4]

        def fx1(epsilon, X4, **kwargs):
          return X4/2 + 0.1*epsilon[5]
        
        def fx2(epsilon, X5, X6, **kwargs):
          return np.exp(X5+X6-10) + epsilon[6]
        
        def fx3(epsilon, X5, X7, **kwargs):
          return np.log(1+(X5+X7)/10) + epsilon[7]

        def fy1(epsilon, X1, X2, U, **kwargs):
          return np.log(1+X1**2) + 2*X2**2 - X1*X2*(U/2) + epsilon[8]**2

        def fy2(epsilon, X2, X3, **kwargs):
          return np.sin(X2**2) - X3**2 - X2*X3 + 50 + epsilon[9]**2


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
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_exploration_sets(self):
        POMIS = [['X1', 'X2', 'X3'], ['X2', 'X3']]
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
        min_intervention = 0
        max_intervention = 5

        dict_ranges = OrderedDict ([
            ('X1', [min_intervention-2, max_intervention]),
            ('X2', [min_intervention, max_intervention]),
            ('X3', [min_intervention, max_intervention]),
            ('X4', [min_intervention-4, max_intervention]),
            ('X5', [min_intervention, max_intervention]),
            ('X6', [min_intervention, max_intervention]),
            ('X7', [min_intervention, max_intervention]),
            ('X8', [min_intervention, max_intervention])
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