import sys
sys.path.append("..") 

import numpy as np
from collections import OrderedDict

from .graph import GraphStructure
from .economics_CostFunctions import define_costs


from ..causal_models.model import CausalDiagram
from ..causal_models.where_do import MISs, bruteforce_POMISs


class Econ(GraphStructure):
    """
    # Observed variables                    
    # Electricity consumption               Total electricity consumption                                                           0.1 billion kWh
    # Economic growth                       GDP                                                                                     0.1 billion yuan
    # Electricity investment                Fixed capital investments for electricity, thermal power, and natural gas supply        0.1 billion yuan
    # Investments in other industries       The difference between total fixed asset investments and power industry investment      0.1 billion yuan
    # Employment                            Total number of people who are employed                                                 10 thousand
    # Development of the secondary industry Outputs of the secondary industry                                                       0.1 billion yuan
    # Development of the tertiary industry  Outputs of the tertiary industry                                                        0.1 billion yuan
    # Proportion of non-agriculture         Sum of proportions of secondary and tertiary economic sectors                           %
    # Labor productivity                    GDP/employment                                                                          Yuan per capita
    # 
    # Latent variables 
    # Energy source structure               Proportion of renewable energy                                                          %
    # Informatization level                 Number of internet users                                                                10 thousand
    #                                       Number of websites                                                                      10 thousand
    # Ecological awareness                  Investment in environmental protection                                                  0.1 billion yuan
    """

    def __init__(self):
      super().__init__()
      X1, X4, X3, X2, X5, X6, X7, X8, Y1, Y2 = 'X1', 'X4', 'X3', 'X2', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2'
      self.G = CausalDiagram({'X1', 'X4', 'X3', 'X2', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2'}, 
                         [(X4, X3), (X4, X2), (X3, X2), (X4, X5), 
                          (X5, X6), (X4, X7),
                          (X4, X8), (X2, X8), (X4, Y1), (X2, Y1),
                          (X8, Y2), (Y1, Y2), (X6, Y2), (X2, Y2)
                          ])
      self.Y = ['Y1', 'Y2']
      self.X = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']


    def define_SEM(self):
        def fX1(epsilon, **kwargs):
        #     # Energy Source Structure
            return 0
        
        def fX4(epsilon, **kwargs):
            # Electricity Consumption
            return 0
    
        def fX3(epsilon, X4, **kwargs):
            # Ecological Awareness
            return 0.889*X4 + 17
        
        def fX2(epsilon, X4, X3, **kwargs):
            # Informatization Level
            return 0.836*X4 + 0.464*X3
        
        def fX5(epsilon, X4, **kwargs):
            # Electricity Investment
            return 0.898*X4
        
        def fX6(epsilon, X5, **kwargs):
            # Investment Other
            return 0.783*X5
        
        def fX7(epsilon, X4, **kwargs):
            # Employment
            return 0.789*X4
        
        def fX8(epsilon, X4, X2, **kwargs):
            # Secondary industry
            return 0.566*X4 + 0.561*X2
        
        def fY1(epsilon, X4, X2, **kwargs):
            # Tertiary industry
            return 0.537*X4 + 0.712*X2
        
        def fY2(epsilon, X8, Y1, X6, X2, **kwargs):
            # Proportion of non-agriculture
            return 0.731*X8 + 0.612*Y1 + 0.662*X6 - 0.605*X2

        graph = OrderedDict([
            ('X1', fX1),
            ('X4', fX4),
            ('X3', fX3),
            ('X2', fX2),
            ('X5', fX5),
            ('X6', fX6),
            ('X7', fX7),
            ('X8', fX8),
            ('Y1', fY1),
            ('Y2', fY2)
        ])
        return graph


    def get_exploration_sets(self):
      mo_cbo = [['X2', 'X4', 'X6']]
      manipulative_variables = [self.X]

      exploration_sets = {
          'mo-cbo': mo_cbo,
          'mobo': manipulative_variables
      }
      return exploration_sets


    def get_interventional_ranges(self):
        dict_ranges = OrderedDict([
            ('X1', [-1, 1]),
            ('X2', [-5, 5]),
            ('X3', [-2, 2]),
            ('X4', [-5, 5]),
            ('X5', [-10, 10]),
            ('X6', [-2, 2]),
            ('X7', [-1, 1])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs