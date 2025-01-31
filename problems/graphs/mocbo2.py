import sys
sys.path.append("..") 

from collections import OrderedDict
import numpy as np

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs



class MO_CBO2(GraphStructure):

    def define_SEM(self):

        def fU(epsilon, **kwargs):
          return np.random.choice([-4, 4], p=[0.5, 0.5])

        def fx4(epsilon, U, **kwargs):
          return epsilon[0]**2 + U

        def fx1(epsilon, X4, **kwargs):
          return X4/2 + epsilon[1]**2
        
        def fx2(epsilon, **kwargs):
          return epsilon[2]**2
        
        def fx3(epsilon, **kwargs):
          return epsilon[3]**2

        def fy1(epsilon, X1, X2, U, **kwargs):
          return np.log(1+X1**2) + 2*X2**2 - X1*X2*(U/2) + epsilon[4]

        def fy2(epsilon, X2, X3, **kwargs):
          return np.sin(X2**2) - X3**2 - X2*X3 + 50 + epsilon[5]


        graph = OrderedDict ([
          ('U', fU),
          ('X4', fx4),
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
        manipulative_variables = [['X1', 'X2', 'X3', 'X4']]

        exploration_sets = {
            'pomis': POMIS,
            'mobo': manipulative_variables
        }
        return exploration_sets
    

    def get_set_MOBO(self):
        manipulative_variables = ['X1', 'X2', 'X3', 'X4']
        return manipulative_variables
    

    def get_interventional_ranges(self):
        dict_ranges = OrderedDict ([
            ('X1', [-2, 5]),
            ('X2', [0, 5]),
            ('X3', [0, 5]),
            ('X4', [-4, 5])
        ])
      
        return dict_ranges
    
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs