import numpy as np

from problems.common import build_problem, calc_causal_pareto_front
from mobo.algorithms import get_algorithm
from visualization.data_export import DataExport
from helpers import *
from utils import *



def Causal_ParetoSelect(args, framework_args, graph, exploration_set, costs, interventional_data):


    ############################  Set up for the Pareto front optimisation
    
    if not os.path.isfile(os.path.join(get_problem_dir(args), 'TrueCausalParetoFront.csv')):
        true_front = calc_causal_pareto_front(args.problem, graph.X)
        write_truefront_csv(args, true_front)

    solution = [None] * len(exploration_set)
    exporter = [None] * len(exploration_set)

    for s, set in enumerate(exploration_set):
        # build problem
        problem = build_problem(args.problem, set)
        args.n_var, args.n_obj = problem.n_var, problem.n_obj
        
        # get initial samples
        if args.mode == 'int_data':
            X_init, Y_init = interventional_data[s][-2:]

        # initialise optimizer
        optimizer = get_algorithm('cps')(problem, set, args.n_iter, None, framework_args)

        # initialise data exporter
        exporter[s] = DataExport(optimizer, X_init, Y_init, args)

        # optimisation
        solution[s] = optimizer.solve(X_init, Y_init)
    
    
    ############################  Loop

    # store info
    hv_next_list = np.zeros((args.n_iter + 1, len(exploration_set)))
    X_next_list = [None] * len(exploration_set)
    Y_next_list = [None] * len(exploration_set)
    current_cost = [0] * len(exploration_set)
    Y_aquisition_list = [0] * len(exploration_set)

    # log file
    experiment_log =  {}
    experiment_log['step'] = [0]
    experiment_log['intervened_set'] = [None]
    experiment_log['relative_hv_improvement'] = [None]
    experiment_log['previous_hv'] = [None]
    experiment_log['current_hv'] = [None]
    experiment_log['cost'] = [0]

    # define intervention function
    target_function_list = [None]*len(exploration_set)
    for s in range(len(exploration_set)):
        target_function_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
                                                model = graph.define_SEM(), targets=graph.Y)


    i = 0
    while np.sum(experiment_log['cost']) < 805:
        print('Optimization step', i)

        # Initialize a new row for this iteration
        hv_next_row = np.zeros(len(exploration_set))

        ## Get new design samples and corresponding performance
        for s in range(len(exploration_set)):

            if i == 0:
                # next batch and the objective functions evaluated
                X_next_list[s], Y_next_list[s], initial_hv, hv_next_list[i+1,s] = next(solution[s])
                hv_next_list[i,s] = initial_hv

            else:
                if s == index:
                    X_next_list[s], Y_next_list[s], _, hv_next_list[i+1,s] = next(solution[s])
                else:
                    X_next_list[s], Y_next_list[s], hv_next_list[i,s], hv_next_list[i+1,s] = X_next_list[s], Y_next_list[s], hv_next_list[i-1,s], hv_next_list[i,s]

            # Calculate cost of the interventions X_next_list[s]
            current_cost[s] = calculate_batch_cost(exploration_set[s],costs,X_next_list[s])
                
            # Relative improvement in hypervolume
            Y_aquisition_list[s] = (hv_next_list[i+1,s] - hv_next_list[i,s])/hv_next_list[i,s]
        
        index = np.where(Y_aquisition_list == np.max(Y_aquisition_list))[0][0]

        # update & export current status to csv
        exporter[index].update(X_next_list[index], Y_next_list[index])
        exporter[index].write_csvs()
            
        # break loop if interventional budget is exhausted
        if (np.sum(experiment_log['cost'])+current_cost[index]) > 805:
            break
        
        # log iteration
        experiment_log['step'].append(i+1)
        experiment_log['intervened_set'].append(get_intervention_set_name(exploration_set[index]))
        experiment_log['relative_hv_improvement'].append(Y_aquisition_list[index])
        experiment_log['previous_hv'].append(hv_next_list[i,index])
        experiment_log['current_hv'].append(hv_next_list[i+1,index])
        experiment_log['cost'].append(current_cost[index])
        save_experiment_log(args, experiment_log)

        # next iteration
        i += 1