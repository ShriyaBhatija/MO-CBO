import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up
import time
import numpy as np
import pandas as pd
import time 

from utils_functions import *

from problems.common import build_problem
from mobo.algorithms import get_algorithm
from visualization.data_export import DataExport
from utils import *



def Causal_ParetoSelect(args, framework_args, graph, exploration_set, costs, functions, full_observational_samples, observational_samples, interventional_data, dict_ranges, data_x_list, data_y_list):

    ## Initialise dicts to store values over trials and assign initial values
    x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set)
    
    ## Define the mean functions and var functions given the current set of observational data. 
    mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var)

    ############################  Set up for the Pareto optimisation
    solution = [None] * len(exploration_set)
    exporter = [None] * len(exploration_set)

    for s, set in enumerate(exploration_set):
        # build problem, get initial samples
        problem, true_pfront, X_init, Y_init = build_problem(args.problem, observational_samples, len(set), len(graph.get_targets()), args.n_init_sample_int, set, args.n_process)
        args.n_var, args.n_obj = problem.n_var, problem.n_obj
       
        if args.mode == 'causal_prior':
            mask = np.ones(len(observational_samples[set[0]]), dtype=bool)
            for var in set:
                lower = dict_ranges[var][0]
                upper = dict_ranges[var][1]
                mask &= (observational_samples[var] >= lower) & (observational_samples[var] <= upper)
            X_init_obs = np.column_stack((observational_samples[var][mask] for var in set))
            Y_init_obs = mean_functions_list[s](X_init_obs)
            # Interventional data
            X_init_int, Y_init_int = interventional_data[s][-2:]
            # Append causal effects from observational data to the interventional data
            X_init = np.concatenate((X_init_obs, X_init_int), axis=0)
            Y_init = np.concatenate((Y_init_obs, Y_init_int), axis=0)
        
        elif args.mode == 'int_data':
            X_init, Y_init = interventional_data[s][-2:]

        # initialize optimizer
        optimizer = get_algorithm(args.algo)(problem, set, args.n_iter, None, framework_args)

        # initialize data exporter
        exporter[s] = DataExport(optimizer, X_init, Y_init, args)

        # optimization
        solution[s] = optimizer.solve(X_init, Y_init)
        print(X_init, Y_init)

        # export true Pareto front to csv
        if s == 0:
            if true_pfront is not None:
                exporter[s].write_truefront_csv(true_pfront)
        
        print('done')
    
        

    #######################################################################################
    ## Loop
    #######################################################################################

    ## Initialise variables
    observed = 0
    trial_intervened = 0.

    ## Define list to store info
    target_function_list = [None]*len(exploration_set)
    model_list = [None]*len(exploration_set)

    # Hypervolumes in each iteration
    hv_next_list = np.zeros((args.n_iter + 1, len(exploration_set)))
    ## Define list to store info
    X_next_list = [None] * len(exploration_set)
    Y_next_list = [None] * len(exploration_set)
    current_cost = [0] * len(exploration_set)
    Y_rel_hv_improvement = [0] * len(exploration_set)
    Y_aquisition_list = [0] * len(exploration_set)

    experiment_log =  {}
    experiment_log['type_trial'] = []
    experiment_log['intervened_set'] = []
    experiment_log['relative_hv_improvement'] = []
    experiment_log['previous_hv'] = []
    experiment_log['current_hv'] = []
    experiment_log['cost'] = []

    ## Define intervention function
    for s in range(len(exploration_set)):
        target_function_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
                                                model = graph.define_SEM(), targets=graph.get_targets())


    ############################# LOOP
    start_time = time.perf_counter()
    #for i in range(args.n_iter):
    i = -1
    while np.sum(experiment_log['cost']) < 800:
        i += 1
        print('Optimization step', i)
        ## Decide to observe or intervene and then recompute the obs coverage
        #coverage_obs = update_hull(observational_samples, manipulative_variables)
        #rescale = observational_samples.shape[0]/max_N
        #epsilon_coverage = (coverage_obs/coverage_total)/rescale

        epsilon_coverage = 0.5

        uniform = np.random.uniform(0.,1.)

        ## At least observe and interve once
        if i == 0:
            uniform = 0.
        if i >= 1:
            uniform = 1.


        if uniform < epsilon_coverage:
            observed += 1

            experiment_log['type_trial'].append(0)
            experiment_log['intervened_set'].append(None)
            experiment_log['relative_hv_improvement'].append(None)
            experiment_log['previous_hv'].append(None)
            experiment_log['current_hv'].append(None)
            experiment_log['cost'].append(0)

            ## Collect observations and append them to the current observational dataset
            new_observational_samples = observe(num_observation = args.batch_size, 
                                        complete_dataset = full_observational_samples, 
                                        initial_num_obs_samples= args.n_init_sample_obs)

            #observational_samples = observational_samples.append(new_observational_samples)
            observational_samples = pd.concat([observational_samples, pd.DataFrame(new_observational_samples)], ignore_index=True)
                
            ## Refit the models for the conditional distributions 
            functions = graph.refit_models(observational_samples)
                
            ## Update the mean functions and var functions given the current set of observational data. This is updating the prior. 
            mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions, 
                                                            observational_samples, x_dict_mean, x_dict_var)


        
        else:
            experiment_log['type_trial'].append(1)
            trial_intervened += 1

            ## When we decid to interve we need to compute the acquisition functions based on the GP models and decide the variable/variables to intervene
            ## together with their interventional data

            ## If in the previous trial we have observed we want to update all the BO models as the mean functions and var functions computed 
            ## via the DO calculus are changed 
            ## If in the previous trial we have intervened we want to update only the BO model for the intervention for which we have collected additional data 
            if experiment_log['type_trial'][i-1] == 0:
                for s in range(len(exploration_set)):
                    #print('Updating model:', s)
                    model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], False)
            else:
                #print('Updating model:', index)
                model_to_update = index
                model_list[index] = update_BO_models(mean_functions_list[index], 
                                                                        var_functions_list[index], 
                                                                        data_x_list[index], data_y_list[index], False)
            

            ## Get new design samples and corresponding performance
            for s in range(len(exploration_set)):

                if i == 1:
                    # next batch and the objective functions evaluated
                    X_next_list[s], Y_next_list[s], initial_hv, hv_next_list[int(trial_intervened),s] = next(solution[s])
                    hv_next_list[0,s] = initial_hv

                else:
                    if s == index:
                        X_next_list[s], Y_next_list[s], _, hv_next_list[int(trial_intervened),s] = next(solution[s])
                    else:
                        X_next_list[s], Y_next_list[s], hv_next_list[int(trial_intervened)-1,s], hv_next_list[int(trial_intervened),s] = X_next_list[s], Y_next_list[s], hv_next_list[int(trial_intervened)-2,s], hv_next_list[int(trial_intervened)-1,s]

                # Calculate cost of the interventions X_next_list[s]
                current_cost[s] = calculate_batch_cost(exploration_set[s],costs, X_next_list[s])
                
                # Relative improvement in hypervolume
                Y_aquisition_list[s] = (hv_next_list[int(trial_intervened),s] - hv_next_list[int(trial_intervened)-1,s])/hv_next_list[int(trial_intervened)-1,s]

                
            index = np.where(Y_aquisition_list == np.max(Y_aquisition_list))[0][0]


            # update & export current status to csv
            exporter[index].update(X_next_list[index], Y_next_list[index])
            exporter[index].write_csvs()

            ## Append the new data and set the new dataset of the BO model 
            data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]], 
                                        [X_next_list[index], Y_next_list[index]])

            data_x_list[index] = np.vstack((data_x_list[index], X_next_list[index])) 
            data_y_list[index] = np.vstack((data_y_list[index], Y_next_list[index]))
            
            model_list[index].set_data(data_x, data_y_x)

            ## Optimise MOBO model given the new data
            model_list[index].optimize()

            # Save the GP model
            # save_model_path = os.path.join(get_result_dir(args), get_intervention_set_name(exploration_set[index]), 'GPy_model')
            # model_list[index].model.save_model(save_model_path)

            ## Which intervention set was intervened on
            if np.sum(experiment_log['cost']) <= 800:
                experiment_log['intervened_set'].append(get_intervention_set_name(exploration_set[index]))
                experiment_log['relative_hv_improvement'].append(Y_aquisition_list[index])
                experiment_log['previous_hv'].append(hv_next_list[int(trial_intervened)-1,index])
                experiment_log['current_hv'].append(hv_next_list[int(trial_intervened),index])
                experiment_log['cost'].append(current_cost[index])
                save_experiment_log(args, experiment_log)


    ## Compute total time for the loop
    total_time = time.perf_counter() - start_time
