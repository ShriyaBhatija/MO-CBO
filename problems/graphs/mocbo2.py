import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from .mocbo2_CostFunctions import define_costs


class MO_CBO2(GraphStructure):
    """
    An instance of the class graph giving the graph structure in the synthetic example 
    
    Parameters
    ----------
    """
    def __init__(self, observational_samples):
        self.A = np.asarray(observational_samples['A'])[:,np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:,np.newaxis]
        self.F = np.asarray(observational_samples['F'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]

    def define_SEM(self):

        def fU1(epsilon, **kwargs):
          return epsilon[0]

        def fU2(epsilon, **kwargs):
          return epsilon[1]

        def fF(epsilon, **kwargs):
          return epsilon[9]

        def fA(epsilon, U1, F, **kwargs):
          return F**2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
          return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
          return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
          return np.exp(-C)/10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
          return A + C + epsilon[6]

        def fY1(epsilon, D, E, U1, U2, **kwargs):
          return (D+E)**2 + U1 + np.exp(-U2) + epsilon[7]
        
        def fY2(epsilon, D, E, **kwargs):
           return (D+E-10)**2+ epsilon[8]

        # This is just a buffer, so the code works for intervention sets of length one
        # Note that this does not influence any other variables (i.e. it is parent and childless)
        def f_control(epsilon, **kwargs):
          return epsilon[6]*0
        

        graph = OrderedDict ([
              ('U1', fU1),
              ('U2', fU2),
              ('F', fF),
              ('A', fA),
              ('B', fB),
              ('C', fC),
              ('D', fD),
              ('E', fE),
              ('Y1', fY1),
              ('Y2', fY2),
              ('control', f_control)
            ])
        return graph


    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_exploration_sets(self):
        MIS = [['B', 'control'], ['D', 'control'], ['E', 'control'], ['B', 'D'], ['B', 'E'], ['D', 'E']]
        POMIS = [['B', 'control'], ['D', 'control'], ['E', 'control'], ['B', 'D'], ['D', 'E']]
        POMIS = [['B', 'D']]
        manipulative_variables = [['B', 'D', 'E']]

        exploration_sets = {
         'mis': MIS,
         'pomis': POMIS,
         'mobo': manipulative_variables
        }
        return exploration_sets



    def get_set_MOBO(self):
        manipulative_variables = ['B', 'D', 'E']
        return manipulative_variables


    def get_interventional_ranges(self):
        min_intervention_e = -1
        max_intervention_e = 1

        min_intervention_b = -4
        max_intervention_b = 1

        min_intervention_d = -5
        max_intervention_d = 5

        min_intervention_f = -4
        max_intervention_f = 4

        min_control = -0.5
        max_control = 0.5

        dict_ranges = OrderedDict ([
          ('E', [min_intervention_e, max_intervention_e]),
          ('B', [min_intervention_b, max_intervention_b]),
          ('D', [min_intervention_d, max_intervention_d]),
          ('F', [min_intervention_f, max_intervention_f]),
          ('control', [min_control, max_control])
        ])
        return dict_ranges
    

    def fit_all_models(self):
       pass

    def refit_models(self, observational_samples):
       pass

    def get_all_do(self):
       pass
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs

