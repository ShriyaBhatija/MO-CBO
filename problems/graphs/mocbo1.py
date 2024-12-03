import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from .mocbo1_CostFunctions import define_costs


class MO_CBO1(GraphStructure):
    
    def __init__(self, observational_samples):
                
        self.X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        self.X3 = np.asarray(observational_samples['X3'])[:,np.newaxis]
        self.X4 = np.asarray(observational_samples['X4'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]
        self.control_node = np.asarray(observational_samples['control'])[:,np.newaxis]

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
        
        # This is just a buffer, so the code works for intervention sets of length one
        # Note that this does not influence any other variables (i.e. it is parent and childless)
        def f_control(epsilon, **kwargs):
          return epsilon[6]*0

        graph = OrderedDict ([
          ('X3', fx3),
          ('X4', fx4),
          ('X1', fx1),
          ('X2', fx2),
          ('Y1', fy1),
          ('Y2', fy2),
          ('control', f_control)
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_exploration_sets(self):
      MIS = [['X1', 'control'], ['X2', 'control'], ['X3', 'control'], ['X4', 'control'], 
             ['X1', 'X2'], ['X3', 'X4'], ['X2', 'X3'], ['X1', 'X4']]
      POMIS = [['X1','X2']]
      manipulative_variables = [['X1', 'X2', 'X3', 'X4']]

      exploration_sets = {
         'mis': MIS,
         'pomis': POMIS,
         'mobo': manipulative_variables
      }
      return exploration_sets
    

    def get_set_MOBO(self):
      manipulative_variables = ['X1', 'X2', 'X3', 'X4']
      return manipulative_variables
    

    def get_interventional_ranges(self):
      min_intervention_x = -1
      max_intervention_x = 2

      min_intervention_z = -1
      max_intervention_z = 3

      min_control = -0.5
      max_control = 0.5

      dict_ranges = OrderedDict ([
          ('X1', [min_intervention_z, max_intervention_z]),
          ('X2', [min_intervention_z, max_intervention_z]),
          ('X3', [min_intervention_x, max_intervention_x]),
          ('X4', [min_intervention_x, max_intervention_x]),
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