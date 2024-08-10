import sys
sys.path.append("..") 

import numpy as np


def compute_do_X1(observational_samples, target_var, functions, value):
  
  gp_Z = functions['gp_X1_toZ1']
  gp_X_Z = functions[f'gp_X1_Z1_to{target_var}']
  
  X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]

  intervened_inputs_X1 = np.repeat(value, X1.shape[0])[:,np.newaxis]
  new_Z = np.mean(gp_Z.predict(intervened_inputs_X1)[0])
 
  intervened_inputs_X_Z = np.hstack((X1, np.repeat(new_Z, X1.shape[0])[:,np.newaxis]))
  
  mean_do = np.mean(gp_X_Z.predict(intervened_inputs_X_Z)[0])
  var_do = np.mean(gp_X_Z.predict(intervened_inputs_X_Z)[1])

  return mean_do, var_do


def compute_do_X2(observational_samples, target_var, functions, value):

  gp_Z = functions['gp_X2_toZ2']
  gp_X_Z = functions[f'gp_X2_Z2_to{target_var}']
  
  X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]

  intervened_inputs_X2 = np.repeat(value, X2.shape[0])[:,np.newaxis]
  new_Z = np.mean(gp_Z.predict(intervened_inputs_X2)[0])

  intervened_inputs_X_Z = np.hstack((X2, np.repeat(new_Z, X2.shape[0])[:,np.newaxis]))
  
  mean_do = np.mean(gp_X_Z.predict(intervened_inputs_X_Z)[0])
  var_do = np.mean(gp_X_Z.predict(intervened_inputs_X_Z)[1])

  return mean_do, var_do


def compute_do_X1X2(observational_samples, target_var, functions, value):
  gp_Z1 = functions[f'gp_X1_toZ1']
  gp_Z2 = functions[f'gp_X2_toZ2']

  gp_X1_X2_Z1_Z2 = functions[f'gp_X1_X2_Z1_Z2_to{target_var}']
  
  X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
  X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]

  intervened_inputs_X1 = np.repeat(value[0], X1.shape[0])[:,np.newaxis]
  new_Z1 = np.mean(gp_Z1.predict(intervened_inputs_X1)[0])

  intervened_inputs_X2 = np.repeat(value[1], X2.shape[0])[:,np.newaxis]
  new_Z2 = np.mean(gp_Z2.predict(intervened_inputs_X2)[0])

  intervened_inputs = np.hstack((X1, X2, np.repeat(new_Z1, X1.shape[0])[:,np.newaxis], np.repeat(new_Z2, X2.shape[0])[:,np.newaxis]))
  
  mean_do = np.mean(gp_X1_X2_Z1_Z2.predict(intervened_inputs)[0])
  var_do = np.mean(gp_X1_X2_Z1_Z2.predict(intervened_inputs)[1])

  return mean_do, var_do


def compute_do_Z1(observational_samples, target_var, functions, value):
  
    gp_Z = functions[f'gp_Z1_to{target_var}']
    
    Z = np.asarray(observational_samples['Z1'])[:,np.newaxis]
    intervened_inputs = np.repeat(value, Z.shape[0])[:,np.newaxis]
    
    mean_do = np.mean(gp_Z.predict(intervened_inputs)[0])
    var_do = np.mean(gp_Z.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_Z2(observational_samples, target_var, functions, value):
  
    gp_Z = functions[f'gp_Z2_to{target_var}']
    
    Z = np.asarray(observational_samples['Z2'])[:,np.newaxis]
    intervened_inputs = np.repeat(value, Z.shape[0])[:,np.newaxis]
    
    mean_do = np.mean(gp_Z.predict(intervened_inputs)[0])
    var_do = np.mean(gp_Z.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_Z1Z2(observational_samples, target_var, functions, value):
  
    gp_Z1_Z2 = functions[f'gp_Z1_Z2_to{target_var}']

    Z1 = np.asarray(observational_samples['Z1'])[:,np.newaxis]
    Z2 = np.asarray(observational_samples['Z2'])[:,np.newaxis]

    intervened_inputs = np.hstack((np.repeat(value[0], Z1.shape[0])[:,np.newaxis], np.repeat(value[1], Z2.shape[0])[:,np.newaxis]))
    
    mean_do = np.mean(gp_Z1_Z2.predict(intervened_inputs)[0])
    var_do = np.mean(gp_Z1_Z2.predict(intervened_inputs)[1])
    
    return mean_do, var_do
