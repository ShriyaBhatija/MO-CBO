## Import basic packages
import numpy as np
import pandas as pd



def sample_from_model(model, epsilon = None):
  if epsilon is None:
     epsilon = np.random.uniform(-1,2,len(model))
     #epsilon = randn(len(model))
  sample = {}
  for variable, function in model.items():
    sample[variable] = function(epsilon, **sample)
  return sample
  

def intervene(*interventions, model):
    new_model = model.copy()

    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions[0].items():
        assign(new_model, variable, value)
  
    return new_model


def compute_target_function(*interventions, model, target_variable, num_samples=1000):
    mutilated_model = intervene(*interventions, model = model)

    samples = [sample_from_model(mutilated_model) for _ in range(num_samples)]
    samples = pd.DataFrame(samples)
    return np.mean(samples[target_variable]), np.var(samples[target_variable])


def intervene_dict(model, **interventions):

    new_model = model.copy()

    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions.items():
        assign(new_model, variable, value)
  
    return new_model


#EDITED to support our multi-objective optimisation framework
def Intervention_function(*interventions, model, targets):
    num_samples = 100

    def compute_target_function_fcn(value):
        num_interventions = len(interventions[0])

        for i in range(num_interventions):
            interventions[0][list(interventions[0].keys())[i]] = value[i]
    
        mutilated_model = intervene_dict(model, **interventions[0])
        np.random.seed(1)
        samples = [sample_from_model(mutilated_model) for _ in range(num_samples)]
        samples = pd.DataFrame(samples)

        result = np.array([np.mean(samples[target]) for target in targets])
        return result

    return compute_target_function_fcn