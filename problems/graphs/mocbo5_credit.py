import sys
sys.path.append("..") 

from collections import OrderedDict
import numpy as np

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs


class SCM_Credit(GraphStructure):

    def define_SEM(self):
        # The equations (with epsilon indices) are:
        #   X1 = U1                                  (Gender, U1 ~ Bernoulli(0.5))
        #   X2 = -35 + U2                            (Age, U2 ~ Gamma(10, 3.5) shifted by -35)
        #   X3 = -0.5 + exp(-(-1 + 0.5 * X1 + (1 + exp(-0.1 * X2)) + U3))
        #   X4 = 1 + 0.01*(X2 - 5)*(5 - X2) + X1 + U4    (Loan Amount)
        #   X5 = -1 + 0.01*X2 + 2*X1 + X4              (Loan Duration; no noise term specified)
        #   X6 = -4 + 0.1*(X2+35) + 2*X1 + X1*X3 + U6    (Income)
        #   X7 = -4 + 1.5 * Piecewise((0, X6 <= 0), (X6, X6 > 0)) + U7   (Savings)
        #   Y  = 1 / (1 + exp(-0.3*(-X4 - X5 + X6 + X7 + X6 * X7)))  (Output probability)
        def fX1(epsilon, **kwargs):
            # Gender
            return epsilon[0]
        
        def fX2(epsilon, **kwargs):
            # Age = -35 + U2
            return -35 + epsilon[1]
        
        def fX3(epsilon, X1, X2, **kwargs):
            # Education = -0.5 + exp(-(-1 + 0.5 * X1 + (1 + anp.exp(-0.1 * X2)) + U3))
            return -0.5 + anp.exp(-(-1 + 0.5 * X1 + (1 + anp.exp(-0.1 * X2)) + epsilon[2]))
        
        def fX4(epsilon, X1, X2, **kwargs):
            # Loan Amount = 1 + 0.01*(X2-5)*(5-X2) + X1 + U4
            return 1 + 0.01 * (X2 - 5) * (5 - X2) + X1 + epsilon[3]
        
        def fX5(epsilon, X1, X2, X4, **kwargs):
            # Loan Duration = -1 + 0.01*X2 + 2*X1 + X4
            # (No explicit noise added; epsilon[4] is ignored or can be added if desired.)
            return -1 + 0.01 * X2 + 2 * X1 + X4
            # Alternatively, add epsilon[4] if a noise term is needed:
            # return -1 + 0.01 * X2 + 2 * X1 + X4 + epsilon[4]
        
        def fX6(epsilon, X1, X2, X3, **kwargs):
            # Income = -4 + 0.1*(X2+35) + 2*X1 + X1*X3 + U6
            return -4 + 0.1 * (X2 + 35) + 2 * X1 + X1 * X3 + epsilon[5]
        
        def fX7(epsilon, X6, **kwargs):
            # Savings = -4 + 1.5 * (0 if X6<=0 else X6) + U7
            return -4 + 1.5 * (anp.where(X6 <= 0, 0, X6)) + epsilon[6]
        
        def fY(epsilon, X4, X5, X6, X7, **kwargs):
            # Output probability = 1 / (1 + exp(-0.3*(-X4 - X5 + X6 + X7 + X6 * X7)))
            return 1 / (1 + anp.exp(-0.3 * (-X4 - X5 + X6 + X7 + X6 * X7)))
            # Note: No epsilon term is added here.
        
        graph = OrderedDict([
            ('X1', fX1),
            ('X2', fX2),
            ('X3', fX3),
            ('X4', fX4),
            ('X5', fX5),
            ('X6', fX6),
            ('X7', fX7),
            ('Y',  fY)
        ])
        return graph

    def get_targets(self):
        # The target variable is the output probability.
        return ['Y']

    def get_exploration_sets(self):
        # TODO: Define exploration sets for the credit SCM.
        exploration_sets = {
            'mo-cbo': [['X1', 'X2']],  # placeholder
            'mobo': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']  # placeholder
        }
        return exploration_sets

    def get_set_MOBO(self):
        # TODO: Define the manipulable variables for the credit SCM.
        return ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']

    def get_interventional_ranges(self):
        # TODO: Specify interventional ranges for each manipulable variable.
        dict_ranges = OrderedDict([
            ('X1', [None, None]),
            ('X2', [None, None]),
            ('X3', [None, None]),
            ('X4', [None, None]),
            ('X5', [None, None]),
            ('X6', [None, None]),
            ('X7', [None, None])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        # TODO: Define the cost structure for the credit SCM.
        costs = define_costs(type_cost)
        return costs