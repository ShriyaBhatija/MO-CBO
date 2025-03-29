import sys
sys.path.append("..") 

from collections import OrderedDict

from .graph import GraphStructure
from .mocbo6_economic_CostFunctions import define_costs
import numpy as np

class SCM_Economics(GraphStructure):
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

    def define_SEM(self):
        def fX1(epsilon, **kwargs):
            # Energy Source Structure
            return np.random.normal(0, 1, 1)[0]
        
        def fX4(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(0, 1, 1)[0]
        
        def fX3(epsilon, X4, **kwargs):
            # Ecological Awareness = 0.889 * X4 + U4
            return 0.889 * X4 + np.random.normal(17, 1, 1)[0]
        
        def fX2(epsilon, X4, X3, **kwargs):
            # Informatization Level = 0.836 * X4 + 0.464 * X3 + U2
            return 0.836 * X4 + 0.464 * X3 + np.random.normal(0, 1, 1)[0]
        
        def fX5(epsilon, X4, **kwargs):
            # Electricity Investment = 0.898 * X4 + U5
            return 0.898 * X4 + np.random.normal(0, 1, 1)[0]
        
        def fX6(epsilon, X5, **kwargs):
            # Investment Other = 0.783 * X5 + U6
            return 0.783 * X5 + np.random.normal(0, 1, 1)[0]
        
        def fX7(epsilon, X4, **kwargs):
            # Employment = 0.789 * X4 + U7
            return 0.789 * X4 + np.random.normal(0, 1, 1)[0]
        
        def fX8(epsilon, X4, X2, **kwargs):
            # Secondary Industry = 0.566 * X4 + 0.561 * X2 + U8
            return 0.566 * X4 + 0.561 * X2 + np.random.normal(0, 1, 1)[0]
        
        def fX9(epsilon, X4, X2, **kwargs):
            # Tertiary Industry = 0.537 * X4 + 0.712 * X2 + U9
            return 0.537 * X4 + 0.712 * X2 + np.random.normal(0, 1, 1)[0]
        
        def fY1(epsilon, X8, X9, X6, X2, **kwargs):
            
            # Prop. non-agriculture = 0.731 * Y1 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + U10
            return 0.731 * X8 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + np.random.normal(0, 1, 1)[0]
        
        def fX11(epsilon, X4, **kwargs):
            # Labor Productivity = 0.918 * X4 + U11
            return 0.918 * X4 + np.random.normal(0, 1, 1)[0]
        
        def fY2(epsilon, X1, X2, X4, X6, X7, Y1, X11, **kwargs):
            # Output threshold = 0.538 * X6 + 0.426 * X7 + 0.826 * X11 + 0.293 * X2 +
            #                    0.527 * X10 + 0.169 * X4 + 0.411 * X1
            return (0.538 * X6 + 0.426 * X7 + 0.826 * X11 +
                    0.293 * X2 + 0.527 * Y1 + 0.169 * X4 + 0.411 * X1)
        
        graph = OrderedDict([
            ('X4', fX4),
            ('X1', fX1),
            ('X3', fX3),
            ('X2', fX2),
            ('X5', fX5),
            ('X6', fX6),
            ('X7', fX7),
            ('X8', fX8),
            ('X9', fX9),
            ('Y1', fY1),
            ('X11', fX11),
            ('Y2',  fY2),
        ])
        return graph

    def get_targets(self):
        return ['Y1', 'Y2']

    def get_exploration_sets(self):
        exploration_sets = {
            'mo-cbo': [['X2', 'X6', 'X11', 'X7'], ['X2', 'X6', 'X9']],  # placeholder
            'mobo': [['X2', 'X9', 'X6', 'X11', 'X7']]  # placeholder
        }
        return exploration_sets

    def get_set_MOBO(self):
        return ['X2', 'X9', 'X6', 'X11', 'X7']

    def get_interventional_ranges(self):
        dict_ranges = OrderedDict([
            ('X2', [0, 100]),
            ('X3', [0, 100]),
            ('X6', [0, 10000000]),
            ('X7', [0, 100]),
            ('X9', [0, 20000]),
            ('X11', [0, 500000])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs
