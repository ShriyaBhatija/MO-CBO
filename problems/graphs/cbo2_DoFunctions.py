import sys
sys.path.append("..") 

import numpy as np



def compute_do_A(observational_samples, target_var, functions, value):
        
    gp_B = functions[f'gp_A_toB']
    gp_A_B = functions[f'gp_A_B_to{target_var}']
        
    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    intervened_inputs_A = np.repeat(value, A.shape[0])[:,np.newaxis]
    new_B = np.mean(gp_B.predict(intervened_inputs_A)[0])

    intervened_inputs_A_B = np.hstack(intervened_inputs_A, np.repeat(new_B, A.shape[0])[:,np.newaxis])
        
    mean_do = np.mean(gp_A_B.predict(intervened_inputs_A_B)[0])
    var_do = np.mean(gp_A_B.predict(intervened_inputs_A_B)[1])

    return mean_do, var_do


def compute_do_B(observational_samples, target_var, functions, value):
  
    gp_B = functions[f'gp_B_to{target_var}']
    
    Z = np.asarray(observational_samples['B'])[:,np.newaxis]
    intervened_inputs = np.repeat(value, Z.shape[0])[:,np.newaxis]
    
    mean_do = np.mean(gp_B.predict(intervened_inputs)[0])
    var_do = np.mean(gp_B.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_AB(observational_samples, target_var, functions, value):
  
    gp_A_B = functions[f'gp_A_B_to{target_var}']

    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    B = np.asarray(observational_samples['B'])[:,np.newaxis]

    intervened_inputs = np.hstack((np.repeat(value[0], A.shape[0])[:,np.newaxis], np.repeat(value[1], B.shape[0])[:,np.newaxis]))
    
    mean_do = np.mean(gp_A_B.predict(intervened_inputs)[0])
    var_do = np.mean(gp_A_B.predict(intervened_inputs)[1])
    
    return mean_do, var_do