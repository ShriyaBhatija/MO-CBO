import sys
sys.path.append("..") 

from collections import OrderedDict
from scipy.stats import truncnorm
import numpy as np

from .graph import GraphStructure
from .health_CostFunctions import define_costs

from ..causal_models.model import CausalDiagram
from ..causal_models.where_do import MISs, bruteforce_POMISs


class Health(GraphStructure):
    def __init__(self):
      super().__init__()
      ci, bmr, height, age, weight, bmi, aspirin, cancer, Y_statin, Y_psa = 'ci', 'bmr', 'height', 'age', 'weight', 'bmi', 'aspirin', 'cancer', 'Y_statin', 'Y_psa'
      self.G = CausalDiagram({'ci', 'bmr', 'height', 'age', 'weight', 'bmi', 'aspirin', 'cancer', 'Y_statin', 'Y_psa'}, 
                         [(bmr, weight), (age, weight), (height, weight), (ci, weight), 
                          (weight, bmi), (height, bmi),
                          (age, aspirin), (bmi, aspirin), (age, Y_statin), (bmi, Y_statin),
                          (age, cancer), (bmi, cancer), (aspirin, cancer), (Y_statin, cancer),
                          (age, Y_psa), (bmi, Y_psa), (aspirin, Y_psa), (Y_statin, Y_psa), (cancer, Y_psa)
                          ])
      self.Y = [Y_statin, Y_psa]
      self.X = [ci, weight, bmi, aspirin]

    
    def define_SEM(self):
        
        def truncated_normal(mean, std, low, upp):
            return truncnorm(
                (low - mean) / std, (upp - mean) / std, loc=mean, scale=std).rvs()
        
        def f_ci(epsilon, **kwargs):
          return np.random.uniform(-100, 100)
        
        def f_bmr(epsilon, **kwargs):
          return 1500 + 10*truncated_normal(0, 1, -0.5, 0.5)
        
        def f_height(epsilon, **kwargs):
          return 175 + 10*truncated_normal(0, 1, -1, 2)

        def f_age(epsilon, **kwargs):
          return np.random.normal(65, 1, 1)[0]
        
        def f_weight(epsilon, bmr, age, height, ci, **kwargs):
          return (bmr + 6.8*age - 5*height) / (13.7 + (ci*150)/7716)
        
        def f_bmi(epsilon, weight, height, **kwargs):
          return weight / (height/100)**2
        
        def f_aspirin(epsilon, age, bmi, **kwargs):
          return 1 / (1 + np.exp(-1*(-8.0 + 0.10*age + 0.03*bmi)))
        
        def f_statin(epsilon, age, bmi, **kwargs):
          return 1 / (1 + np.exp(-1*(-13.0 + 0.10*age + 0.20*bmi)))

        def f_cancer(epsilon, age, bmi, aspirin, Y_statin, **kwargs):
          return 1 / (1 + np.exp(-1*(2.2 - 0.05*age + 0.01*bmi - 0.04*Y_statin + 0.02*aspirin)))

        def f_psa(epsilon, age, bmi, aspirin, Y_statin, cancer, **kwargs):
          return 6.8 + 0.04*age - 0.15*bmi - 0.60*Y_statin + 0.55*aspirin + 1.0*cancer + np.random.normal(0, 0.4, 1)[0]
        

        graph = OrderedDict ([
          ('ci', f_ci),
          ('bmr', f_bmr),
          ('height', f_height),
          ('age', f_age),
          ('weight', f_weight),
          ('bmi', f_bmi),
          ('Y_statin', f_statin),
          ('aspirin', f_aspirin),
          ('cancer', f_cancer),
          ('Y_psa', f_psa)
            ])

        return graph
    

    def get_exploration_sets(self):
      mo_cbo = [['bmi', 'aspirin']]
      manipulative_variables = [self.X]

      exploration_sets = {
          'mo-cbo': mo_cbo,
          'mobo': manipulative_variables
      }
      return exploration_sets
    

    def get_interventional_ranges(self):

      dict_ranges = OrderedDict ([
          ('bmi', [20, 30]),
          ('weight', [50, 100]),
          ('ci', [-100, 100]),
          ('aspirin', [0, 1.0]),
        ])
      
      return dict_ranges
    
    
    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs