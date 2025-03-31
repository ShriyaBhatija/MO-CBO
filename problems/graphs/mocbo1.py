import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp

from .graph import GraphStructure
from .mocbo1_CostFunctions import define_costs

from ..causal_models.model import CausalDiagram
from ..causal_models.where_do import MISs, bruteforce_POMISs

class MO_CBO1(GraphStructure):
    def __init__(self):
      super().__init__()
      X1, X2, X3, X4, Y1, Y2 = 'X1', 'X2', 'X3', 'X4', 'Y1', 'Y2'
      self.G = CausalDiagram({'X1', 'X2', 'X3', 'X4', 'Y1', 'Y2'}, 
                         [(X4, X1), (X4, X2), (X3, X1), (X3, X2), (X2, Y1), (X2, Y2), (X1, Y1), (X1, Y2)])
      self.Y = sorted([node for node in self.G.V if node.startswith('Y')])
      self.X = sorted([node for node in self.G.V if not node.startswith('Y')])


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
  

    def get_exploration_sets(self):
      mo_cbo = bruteforce_POMISs(self.G, self.Y)
      mo_cbo = [sorted(list(set)) for set in mo_cbo]

      mis = MISs(self.G, self.Y)
      mis = [sorted(list(set)) for set in mis]

      manipulative_variables = [self.X]

      exploration_sets = {
          'mo-cbo': mo_cbo,
          'mis': mis,
          'mobo': manipulative_variables
      }
      return exploration_sets
    

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