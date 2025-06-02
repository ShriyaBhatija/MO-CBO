import sys
sys.path.append("..") 

import numpy as np
from collections import OrderedDict

from .graph import GraphStructure
from .credit_CostFunctions import define_costs


from ..causal_models.model import CausalDiagram


class Credit(GraphStructure):
    def __init__(self):
      super().__init__()
      X1, X2, X3, X4, Y1, X6, X7, Y2 = 'X1', 'X2', 'X3', 'X4', 'Y1', 'X6', 'X7', 'Y2'
      self.G = CausalDiagram({'X1', 'X2', 'X3', 'X4', 'Y1', 'X6', 'X7', 'Y2'}, 
                         [(X1, X3), (X2, X3),
                          (X1, X4), (X2, X4),
                          (X1, Y1), (X2, Y1), (X4, Y1),
                          (X1, X6), (X2, X6), (X3, X6),
                          (X6,X7),
                          (X4, Y2), (Y1, Y2), (X6, Y2), (X1, Y2)
                          ])
      self.Y = ['Y1', 'Y2']
      self.X = ['X3', 'X4', 'X6', 'X7']


    def define_SEM(self):
        def fX1(epsilon, **kwargs):
            return 0
        
        def fX2(epsilon, **kwargs):
            return -35 + 35 + 0*np.random.gamma(350,0.1,None)
    
        def fX3(epsilon, X1, X2, **kwargs):
            return -0.5 + 1./(1. + np.exp(-(-1+0.5*X1 + (1./(1+np.exp(-0.1*X2))) + np.random.normal(0, 0.25, 1)[0] )))
        
        def fX4(epsilon, X1, X2, **kwargs):
            return 1 + 0.01*(X2-5)*(5-X2) + X1 + np.random.normal(0, 1, 1)[0]
        
        def fY1(epsilon, X1, X2, X4, **kwargs):
            return -1*(-1 + 0.01*X2 + 2*X1 + X4) + np.random.normal(0, 2, 1)[0]
        
        def fX6(epsilon, X1, X2, X3, **kwargs):
            return -4 + 0.1*(X2 + 35) + 2*X1 + X1*X3 + np.random.normal(0, 1, 1)[0]
        
        def fX7(epsilon, X6, **kwargs):
            return -4 + int(X6 > 0)*X6 + 0*np.random.normal(0, 5, 1)[0]
        
        def fY2(epsilon, X4, Y1, X6, X7, **kwargs):
            return -(1./(1 + np.exp(-0.3*(-X4+Y1+X6+X7+X6*X7))))
    

        graph = OrderedDict([
            ('X1', fX1),
            ('X2', fX2),
            ('X3', fX3),
            ('X4', fX4),
            ('Y1', fY1),
            ('X6', fX6),
            ('X7', fX7),
            ('Y2', fY2)
        ])
        return graph


    def get_exploration_sets(self):
      mo_cbo = [['X4', 'X6', 'X7']]
      manipulative_variables = [self.X]

      exploration_sets = {
          'mo-cbo': mo_cbo,
          'mobo': manipulative_variables
      }
      return exploration_sets


    def get_interventional_ranges(self):
        dict_ranges = OrderedDict([
            ('X3', [-0.5,0.5]),
            ('X4', [-1, 2]),
            ('X6', [-2, 1]),
            ('X7', [-5, 1])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs