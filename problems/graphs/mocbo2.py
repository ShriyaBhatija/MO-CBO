import sys
sys.path.append("..") 

from collections import OrderedDict
import numpy as np

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs

from ..causal_models.model import CausalDiagram
from ..causal_models.where_do import MISs, bruteforce_POMISs

class MO_CBO2(GraphStructure):
    def __init__(self):
      super().__init__()

      X1, X2, X3, X4, Y1, Y2 = 'X1', 'X2', 'X3', 'X4', 'Y1', 'Y2'
      U_X4Y1 = 'U'

      self.G = CausalDiagram({'X1', 'X2', 'X3', 'X4', 'Y1', 'Y2'}, [(X4, X1), (X3, Y2), (X2, Y2), (X1, Y1)], [(X4, Y1, U_X4Y1)])
      self.Y = sorted([node for node in self.G.V if node.startswith('Y')])
      self.X = sorted([node for node in self.G.V if not node.startswith('Y')])


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
          return np.log(1+X1**2) + 2*X2**2 - 0*X1*X2*(U/2) + 0*epsilon[4]

        def fy2(epsilon, X2, X3, **kwargs):
          return np.sin(X2**2) - X3**2 - X2*X3 + 50 + 0*epsilon[5]


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
    

    def get_exploration_sets(self):
      mo_cbo = bruteforce_POMISs(self.G, self.Y)
      mo_cbo = sorted([sorted(list(set)) for set in mo_cbo])

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