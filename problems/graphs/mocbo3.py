import sys
sys.path.append("..") 

from collections import OrderedDict
import numpy as np

from .graph import GraphStructure
from .mocbo3_CostFunctions import define_costs


class MO_CBO3(GraphStructure):

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
        

        graph = OrderedDict ([
          ('U', fU),
          ('X1', fx1),
          ('X2', fx2),
          ('X3', fx3),
          ('Y1', fy1),
          ('Y2', fy2),
          ('Y3', fy3)
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2', 'Y3'] 
    

    def get_exploration_sets(self):
      mo_cbo = [['X1','X2'], ['X1', 'X2', 'X3']]
      manipulative_variables = [['X1', 'X2', 'X3']]

      exploration_sets = {
         'mo-cbo': mo_cbo,
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
          ('X3', [-5, 5])
            ])
      
      return dict_ranges

    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs