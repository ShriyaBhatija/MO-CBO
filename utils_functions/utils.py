## Import basic packages
import numpy as np

#from GPy.kern import RBF
#from GPy.models.gp_regression import GPRegression



def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict


def initialise_dicts(exploration_set):
    current_best_x = {}
    current_best_y = {}
    x_dict_mean = {}
    x_dict_var = {}
    dict_interventions = []

    for i in range(len(exploration_set)):
      variables = exploration_set[i]
      if len(variables) == 1:
        variables = variables[0]
      if len(variables) > 1:
        num_var = len(variables)
        string = ''
        for j in range(num_var):
          string += variables[j]
        variables = string

      dict_interventions.append(variables)

      current_best_x[variables] = []
      current_best_y[variables] = []

      x_dict_mean[variables] = {}
      x_dict_var[variables] = {}
      
    return x_dict_mean, x_dict_var, dict_interventions


def add_data(original, new):
    data_x = np.append(original[0], new[0], axis=0)
    data_y = np.append(original[1], new[1], axis=0)
    return data_x, data_y  


'''
def fit_single_GP_model(X, Y, parameter_list, ard = False):
    kernel = RBF(X.shape[1], ARD = parameter_list[3], lengthscale=parameter_list[0], variance = parameter_list[1]) 
    gp = GPRegression(X = X, Y = Y, kernel = kernel, noise_var= parameter_list[2])
    gp.likelihood.variance.fix(1e-2)
    gp.optimize()
    return gp
'''
