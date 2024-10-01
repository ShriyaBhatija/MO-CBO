## Import basic packages
import numpy as np
import copy


def get_do_function_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        string += str(intervention_variables[i]) 
    total_string = 'compute_do_' + string
    return total_string


def update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var):
    print(dict_interventions)
    mean_functions_list = []
    var_functions_list = []

    for j in range(len(exploration_set)):
        mean_functions_list.append(update_mean_fun(graph, functions, dict_interventions[j], observational_samples, x_dict_mean))
        var_functions_list.append(update_var_fun(graph, functions, dict_interventions[j], observational_samples, x_dict_var))
    return mean_functions_list, var_functions_list


#EDITED to support multi-objective optimisation
def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean):

    def compute_mean(num_interventions, x, xi_dict_mean, compute_do):
        targets = graph.get_targets()
        mean_do = np.zeros((num_interventions, len(targets)))

        for  index, target in enumerate(targets):
            # Create a copy of the dictionary to handle multiple targets
            xi_dict_mean_copy = copy.deepcopy(xi_dict_mean)
            for i in range(num_interventions):
                xi_str = str(x[i])
                if xi_str in xi_dict_mean_copy:
                    mean_do[i,index] = xi_dict_mean_copy[xi_str]
                else:
                    mean_do[i,index], _ = compute_do(observational_samples, target, functions, value = x[i])
                    xi_dict_mean_copy[xi_str] = mean_do[i,index]

        return mean_do

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = compute_mean(num_interventions, x, xi_dict_mean[variables], do_functions[function_name])
        return np.float64(mean_do)

    return mean_function_do


#EDITED to support multi-objective optimisation
def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var):

    def compute_var(num_interventions, x, xi_dict_var, compute_do):
        targets = graph.get_targets()
        var_do = np.zeros((num_interventions, len(targets)))

        for index, target in enumerate(targets):
            for i in range(num_interventions):
                xi_str = str(x[i])
                if xi_str in xi_dict_var:
                    var_do[i,index] = xi_dict_var[xi_str]
                else:
                    _, var_do[i,index] = compute_do(observational_samples, target, functions, value = x[i])
                    xi_dict_var[xi_str] = var_do[i,index]

        return var_do

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def var_function_do(x):
        num_interventions = x.shape[0]    
        var_do = compute_var(num_interventions, x, xi_dict_var[variables], do_functions[function_name])
        return np.float64(var_do)

    return var_function_do  