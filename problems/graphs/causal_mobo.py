import autograd.numpy as anp
import numpy as np
import pandas as pd

from problems.problem import Problem
from utils_functions.graph_functions import Intervention_function
from utils_functions.utils import get_interventional_dict

class CausalMOBO(Problem):
  def __init__(self, graph, intervention_set, **kwargs):
    super().__init__(n_var=len(intervention_set), 
                    n_obj=len(graph.get_targets()), 
                    n_constr=0, 
                    xl=[graph.get_interventional_ranges()[variable][0] for variable in intervention_set], 
                    xu=[graph.get_interventional_ranges()[variable][1] for variable in intervention_set], 
                    type_var=anp.double,
                    **kwargs
                    )
    
    self.intervention_set = intervention_set
    self.graph = graph
  

  def _evaluate_F(self, x):
  # Evaluates in batches

    interventions = {}
    for i, variable in enumerate(self.intervention_set):
      interventions[variable] = x[:,i]

    target_function = Intervention_function(get_interventional_dict(self.intervention_set),
									model = self.graph.define_SEM(), targets = self.graph.get_targets())
            
    f = []
    for i in range(x.shape[0]):
        target_value = target_function(x[i])
        f.append(target_value)
    return np.asarray(f)
