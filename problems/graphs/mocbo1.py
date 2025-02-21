import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp

from .graph import GraphStructure
from .mocbo1_CostFunctions import define_costs


class MO_CBO1(GraphStructure):

    def define_SEM(self):

        def fx3(epsilon, **kwargs):
          return epsilon[0]
        
        def fx4(epsilon, **kwargs):
          return epsilon[1]
        
        def fx1(epsilon, X3, X4, **kwargs):
          return ((X3-X4)/2)**3 + epsilon[2]
        
        def fx2(epsilon, X3, X4, **kwargs):
          return anp.exp((X3-X4)/2) + epsilon[3]

        def fy1(epsilon, X1, X2, **kwargs):
          return (X1+X2)**2 + epsilon[4]

        def fy2(epsilon, X1, X2, **kwargs):
          return (X1+X2-10)**2 + epsilon[5]
        

        graph = OrderedDict ([
          ('X3', fx3),
          ('X4', fx4),
          ('X1', fx1),
          ('X2', fx2),
          ('Y1', fy1),
          ('Y2', fy2)
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_exploration_sets(self):
      mo_cbo = [['X1','X2']]
      manipulative_variables = [['X1', 'X2', 'X3', 'X4']]

      exploration_sets = {
         'mo-cbo': mo_cbo,
         'mobo': manipulative_variables
      }
      return exploration_sets
    

    def get_set_MOBO(self):
      manipulative_variables = ['X1', 'X2', 'X3', 'X4']
      return manipulative_variables
    

    def get_interventional_ranges(self):
      min_intervention_x = -1
      max_intervention_x = 1

      min_intervention_z = -1
      max_intervention_z = 2

      dict_ranges = OrderedDict ([
          ('X1', [min_intervention_z, max_intervention_z]),
          ('X2', [min_intervention_z, max_intervention_z]),
          ('X3', [min_intervention_x, max_intervention_x]),
          ('X4', [min_intervention_x, max_intervention_x])
        ])
      
      return dict_ranges  
    
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs