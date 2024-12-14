import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from .health_CostFunctions import define_costs


class Health(GraphStructure):
    
    def __init__(self, observational_samples):
                
        self.age = np.asarray(observational_samples['age'])[:,np.newaxis]
        self.bmi = np.asarray(observational_samples['bmi'])[:,np.newaxis]
        self.statin = np.asarray(observational_samples['statin'])[:,np.newaxis]
        self.aspirin = np.asarray(observational_samples['aspirin'])[:,np.newaxis]
        self.cancer = np.asarray(observational_samples['cancer'])[:,np.newaxis]
        self.psa = np.asarray(observational_samples['psa'])[:,np.newaxis]
        self.control_node = np.asarray(observational_samples['control'])[:,np.newaxis]

    def define_SEM(self):

        def f_age(epsilon, **kwargs):
          return np.random.normal(65, 0.5, 1)[0]
        
        def f_bmi(epsilon, age, **kwargs):
          return 27.0 - 0.01*age + np.random.normal(0, 0.7, 1)[0]
        
        def f_aspirin(epsilon, age, bmi, **kwargs):
          return 1 / (1 + np.exp(-1*(-8.0 + 0.10*age + 0.03*bmi)))
        
        def f_statin(epsilon, age, bmi, **kwargs):
          return 1 / (1 + np.exp(-1*(-13.0 + 0.10*age + 0.20*bmi)))

        def f_cancer(epsilon, age, bmi, aspirin, statin, **kwargs):
          return 1 / (1 + np.exp(-1*(2.2 - 0.05*age + 0.01*bmi - 0.04*statin + 0.02*aspirin)))

        def f_psa(epsilon, age, bmi, aspirin, statin, cancer, **kwargs):
          return 6.8 + 0.04*age - 0.15*bmi - 0.60*statin + 0.55*aspirin + 1.0*cancer + np.random.normal(0, 0.4, 1)[0]
        
        # This is just a buffer, so the code works for intervention sets of length one
        # Note that this does not influence any other variables (i.e. it is parent and childless)
        def f_control(epsilon, **kwargs):
          return epsilon[6]*0

        graph = OrderedDict ([
          ('age', f_age),
          ('bmi', f_bmi),
          ('statin', f_statin),
          ('aspirin', f_aspirin),
          ('cancer', f_cancer),
          ('psa', f_psa),
          ('control', f_control)
        ])

        return graph
    

    def get_targets(self):
        return ['cancer', 'psa'] 
    

    def get_exploration_sets(self):
      MIS = [['statin','control'], ['aspirin', 'control'], ['bmi', 'control'],
              ['statin', 'aspirin'], ['statin', 'bmi'], ['aspirin', 'bmi'],
              ['aspirin', 'statin', 'bmi']
              ]
      POMIS = [['aspirin', 'statin', 'bmi']]
      manipulative_variables = [['aspirin', 'statin', 'bmi']]

      exploration_sets = {
          'mis': MIS,
          'pomis': POMIS,
          'mobo': manipulative_variables
      }
      return exploration_sets
    

    def get_set_MOBO(self):
      manipulative_variables = ['aspirin', 'statin', 'bmi']
      return manipulative_variables
    

    def get_interventional_ranges(self):

      dict_ranges = OrderedDict ([
          ('bmi', [20, 30]),
          ('statin', [0, 1.0]),
          ('aspirin', [0, 1.0]),
          ('control', [-0.5, 0.5])
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