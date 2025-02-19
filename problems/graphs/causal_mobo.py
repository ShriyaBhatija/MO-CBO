import autograd.numpy as anp

from problems.problem import Problem
from helpers.graph_functions import Intervention_function
from helpers.utils import get_interventional_dict


'''
Custom Pymoo Problem class for causal multi-objective optimisation problems.
'''
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
    '''
    Evaluate the local mo-cbo problems in batches.
    '''
    interventions = {}
    for i, variable in enumerate(self.intervention_set):
      interventions[variable] = x[:,i]

    target_function = Intervention_function(get_interventional_dict(self.intervention_set),
									model = self.graph.define_SEM(), targets = self.graph.get_targets())
            
    f = []
    for i in range(x.shape[0]):
        target_value = target_function(x[i])
        f.append(target_value)
    return anp.asarray(f)
  

  def _calc_pareto_front(self, n_pareto_points=100):
    '''
    Calculate the true causal Pareto front by evaluating the target function for each intervention set 
    and filtering the Pareto optimal points
    '''
    def is_pareto_optimal(points):
      is_efficient = anp.ones(points.shape[0], dtype = bool)
      for i, c in enumerate(points):
        if is_efficient[i]:
          is_efficient[is_efficient] = anp.any(points[is_efficient]<c, axis=1) 
          is_efficient[i] = True 
      return is_efficient
    
    if 'mo-cbo' in self.graph.get_exploration_sets():
      exploration_set = self.graph.get_exploration_sets()['mo-cbo']
    elif 'mis' in self.graph.get_exploration_sets():
      exploration_set = self.graph.get_exploration_sets()['mis']
    else:
      exploration_set = self.graph.get_exploration_sets()['mobo']

    f = []

    for set in exploration_set:
      xl =[self.graph.get_interventional_ranges()[variable][0] for variable in set]
      xu =[self.graph.get_interventional_ranges()[variable][1] for variable in set] 

      ranges = [anp.linspace(xl[i], xu[i], n_pareto_points) for i in range(len(set))]
      grids = anp.meshgrid(*ranges)
      points = anp.vstack([grid.ravel() for grid in grids]).T

      target_function = Intervention_function(get_interventional_dict(set),
									model = self.graph.define_SEM(), targets = self.graph.get_targets(), num_samples=1)
    
      for i in range(points.shape[0]):
        print(f'{i}/{points.shape[0]}')
        target_value = target_function(points[i])
        f.append(target_value)

    pareto_points = is_pareto_optimal(anp.array(f))
    filtered_points = [value for value, flag in zip(f, pareto_points) if flag]

    return anp.array(filtered_points)
