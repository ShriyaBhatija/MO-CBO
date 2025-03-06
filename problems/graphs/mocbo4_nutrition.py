from collections import OrderedDict
import sys
sys.path.append("..") 

from collections import OrderedDict
import numpy as np

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs

class SCM_Nutrition(GraphStructure):

    def define_SEM(self):
        # The order here corresponds to the epsilon vector indices.
        # Equations:
        #   X1 = U1          (Age, where U1 ~ N(25,1)) 
        #   X2 = U2          (Sex, where U2 ~ N(0,1))
        #   X3 = 0.02 * X1 + U3      (Blood Pressure)
        #   X4 = 0.12 * X3 + U4      (SBP)
        #   X5 = 0.02 * X4 + U5      (Pulse Pressure)
        #   X6 = U6          (Inflammation)
        #   X7 = U7          (Poverty Index)
        #   X8 = 0.03 * X7 + U8      (Sedimentation RAE)
        #   y  = -0.21 * X2 - 0.59 * X1 + 0.03 * X8 - 0.04 * X7 + 0.02 * X5 + 0.1 * X4
        def fX1(epsilon, **kwargs):
            # Age
            # TODO: Incorporate constant if needed (e.g., shift noise to mean 25)
            return epsilon[0]
        
        def fX2(epsilon, **kwargs):
            # Sex
            # TODO: Adjust noise for distribution N(0,1) if required
            return epsilon[1]
        
        def fX3(epsilon, X1, **kwargs):
            # Blood Pressure = 0.02 * X1 + U3
            return 0.02 * X1 + epsilon[2]
        
        def fX4(epsilon, X3, **kwargs):
            # SBP = 0.12 * X3 + U4
            return 0.12 * X3 + epsilon[3]
        
        def fX5(epsilon, X4, **kwargs):
            # Pulse Pressure = 0.02 * X4 + U5
            return 0.02 * X4 + epsilon[4]
        
        def fX6(epsilon, **kwargs):
            # Inflammation = U6
            return epsilon[5]
        
        def fX7(epsilon, **kwargs):
            # Poverty Index = U7
            return epsilon[6]
        
        def fX8(epsilon, X7, **kwargs):
            # Sedimentation RAE = 0.03 * X7 + U8
            return 0.03 * X7 + epsilon[7]
        
        def fy(epsilon, X1, X2, X4, X5, X7, X8, **kwargs):
            # Risk = -0.21 * X2 - 0.59 * X1 + 0.03 * X8 - 0.04 * X7 + 0.02 * X5 + 0.1 * X4
            # (No explicit noise term is given; if desired, add epsilon[8])
            return -0.21 * X2 - 0.59 * X1 + 0.03 * X8 - 0.04 * X7 + 0.02 * X5 + 0.1 * X4  # + epsilon[8] if noise is needed
        
        graph = OrderedDict([
            ('X1', fX1),
            ('X2', fX2),
            ('X3', fX3),
            ('X4', fX4),
            ('X5', fX5),
            ('X6', fX6),
            ('X7', fX7),
            ('X8', fX8),
            ('y',  fy)
        ])
        return graph

    def get_targets(self):
        # The target variable is the risk outcome.
        return ['y']

    def get_exploration_sets(self):
        # TODO: Define exploration sets (e.g., mo-cbo, mobo) based on domain knowledge.
        exploration_sets = {
            'mo-cbo': [['X1', 'X2']],  # TODO: example placeholder
            'mobo': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']  # TODO: example placeholder
        }
        return exploration_sets

    def get_set_MOBO(self):
        # TODO: Define which variables are manipulable.
        return ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

    def get_interventional_ranges(self):
        # TODO: Provide the intervention ranges for each manipulable variable.
        dict_ranges = OrderedDict([
            ('X1', [0, 100]),  # e.g., [min, max]
            ('X2', [0, 1]),
            ('X3', [60, 180]),
            ('X4', [60, 180]),
            ('X5', [0, 120]),
            ('X6', [None, None]),
            ('X7', [None, None]),
            ('X8', [None, None])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        # TODO: Define cost structure details for the nutrition SCM.
        costs = define_costs(type_cost)  # Assuming define_costs is defined elsewhere.
        return costs