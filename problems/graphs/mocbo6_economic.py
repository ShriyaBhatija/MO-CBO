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
        # def fX1(epsilon, **kwargs):
        #     # Energy Source Structure
        #     return np.random.normal(0, 1, 1)[0]
        
        def fU1(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU2(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU4(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU5(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU6(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU7(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU8(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        
        def fU9(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU10(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
        
        def fU11(epsilon, **kwargs):
            # Electricity Consumption
            return np.random.normal(1, 1, 1)[0]
    
        def fX3(epsilon, U4, **kwargs):
            # Ecological Awareness: X3 = 0.889 * X4 + U4
            return 0.889 * U4 + U4
    
        def fX2(epsilon, U2, U4,X3, **kwargs):
            # Informatization Level = 0.836 * U4 + 0.464 * X3 + U2
            return  0.836 * U4 + 0.464 * X3 + U2
    
        def fX5(epsilon, U1, U4 ,U5, **kwargs):
            return  0.898 * U4 + U5
        
        def fX6(epsilon, X5, U6, **kwargs):
            # Investment Other = 0.783 * X5 + U6
            return  0.783 * X5 + U6

        def fY1(epsilon, X5, X6, **kwargs):
            # Total Investment
            return  X5 + X6
        
        def fX10(epsilon, U1, X6, C1, X2, Y1,  **kwargs):
            # Prop. non-agriculture = 0.731 * Y1 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + U10
            return 0.731 * X8 + 0.612 * Y1 + 0.662 * X6 + 0.605 * X2 + U10
        
        def fY2(epsilon, X2, U4, X6, U1, U7, U8, U9, U10, U11, **kwargs):
            # Output threshold = 0.538 * X6 + 0.426 * X7 + 0.826 * X11 + 0.293 * X2 +
            #                    0.527 * X10 + 0.169 * U1 + 0.411 * X1
            return (0.538 * X6 + 0.426 * (0.789 * U4 + U7) + 0.826 * (0.918 * U4 + U11) + 0.293 * X2 + 
            0.527 * (0.731 * (0.566 * U4 + 0.561 * X2 + U8) + 0.612 * (0.537 * U4 + 0.712 * X2 + U9) + 0.662 * X6 + 0.605 * X2 + U10) + 0.169 * U4 + 0.411 * U1)
        
        graph = OrderedDict([
            ('U1', fU1),
            ('U2', fU2),
            ('U4', fU4),
            ('U5', fU5),
            ('U6', fU6),
            ('U7', fU7),
            ('U8', fU8),
            ('U9', fU9),
            ('U10', fU10),
            ('U11', fU11),
            ('X3', fX3),
            ('X2', fX2),
            ('X5', fX5),
            ('X6', fX6),
            ('Y1', fY1),
            ('Y2', fY2),
        ])
        return graph

    def get_targets(self):
        return ['Y1', 'Y2']

    def get_exploration_sets(self):
        exploration_sets = {
            'mo-cbo': [['X5', 'X6'], ['X2', 'X6']],  # placeholder
            'mobo': [['X2', 'X5', 'X6']]  # placeholder
        }
        return exploration_sets

    def get_set_MOBO(self):
        return ['X2', 'X5', 'X6']

    def get_interventional_ranges(self):
        # Define the equations
        # scm_economics_equations = [
        #     #Eq(X4, U4),                                   # Electricity Cons.: X4 = U4 ~ N(0,100000)
        #     Eq(X3, 0.889 * U4 + U4),                      # Ecological Awareness: X3 = 0.889 * X4 + U4
        #     Eq(X2, 0.836 * X4 + 0.464 * X3 + U2),         # Informatization Level: X2 = 0.836 * X4 + 0.464 * X3 + U2
        #     Eq(X5, 0.898 * X4 + U5),                      # Electricity Investment: X5 = 0.898 * X4 + U5
        #     Eq(X6, 0.783 * X5 + U6),                      # Investment Other: X6 = 0.783 * X5 + U6
        #     Eq(Y1, X5 + X6),                               # Total Investment: X5 = 0.898 * X4 + U5
        #     Eq(Y2, (0.538 * X6 + 0.426 * (0.789 * X4 + U7) + 0.826 * (0.918 * X4 + U11) + 0.293 * X2 + 
        #         0.527 * (0.731 * (0.566 * X4 + 0.561 * X2 + U8) + 0.612 * (0.537 * X4 + 0.712 * X2 + U9) + 0.662 * X6 + 0.605 * X2 + U10) + 0.169 * X4 + 0.411 * U1))  # Output threshold
        # ]
        dict_ranges = OrderedDict([
            ('X2', [0, 100]),
            ('X5', [0, 100]),
            ('X6', [0, 100]),
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs
