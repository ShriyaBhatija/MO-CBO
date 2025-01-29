from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from utils import get_problem_dir
from arguments import get_vis_args
from metrics import generational_distance, is_pareto_optimal



def calculate_metrics(args, problem_dir, true_front, exp_set, n_targets):
    args.exp_set = exp_set

    # Determine the minimum number of iterations across all seeds
    min_n_iter = np.inf
    for seed in range(0, 10):
        args.seed = seed
        experiment_log = pd.read_csv(f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/' + 'experiment_log.csv')[1:]
        min_n_iter = min(min_n_iter, len(experiment_log))

    # Initialize arrays
    gd = np.zeros((min_n_iter+1, 10))
    igd = np.zeros((min_n_iter+1, 10))
    costs = np.zeros((min_n_iter+1, 10))

    for seed in range(0,10):
        args.seed = seed
        cost = 0

        count: Dict[str, int] = {}
        all_pareto_points = []

        experiment_log = pd.read_csv(f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/' + 'experiment_log.csv')

        for iterID in range(0, min_n_iter+1):

            if iterID == 0:
                 # Before any iteration, go over all sets and get the initial Pareto points
                intervention_sets = set(np.asarray((experiment_log['intervened_set']))[1:])
                for i_set in intervention_sets:
                    csv_folder = f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/{i_set}/'
                    paretoEval = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
                    # Get the points from the Pareto front of the last iteration (i.e. the complete approximation)
                    points = paretoEval[paretoEval['iterID'] == iterID]
                    for _, row in points.iterrows():
                        if n_targets == 2:
                            all_pareto_points.append([row['Pareto_f1'], row['Pareto_f2']])
                        elif n_targets == 3:
                            all_pareto_points.append([row['Pareto_f1'], row['Pareto_f2'], row['Pareto_f3']])

            else:
                intervention_set = np.asarray((experiment_log['intervened_set']))[iterID]
                count[intervention_set] = count.get(intervention_set, 0) + 1

                for i_set, n in count.items():
                    csv_folder = f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/{i_set}/'
                    paretoEval = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
            
                    # Get the points from the Pareto front of the last iteration (i.e. the complete approximation)
                    points = paretoEval[paretoEval['iterID'] == n]
                    for _, row in points.iterrows():
                        if n_targets == 2:
                            all_pareto_points.append([row['Pareto_f1'], row['Pareto_f2']])
                        elif n_targets == 3:
                            all_pareto_points.append([row['Pareto_f1'], row['Pareto_f2'], row['Pareto_f3']])
            
            # Calculate Pareto efficient points
            pareto_flags = is_pareto_optimal(np.array([point for point in all_pareto_points]))
            pareto_points = np.asarray([value for value, flag in zip(all_pareto_points, pareto_flags) if flag])

            gd[iterID,seed] = generational_distance(pareto_points, true_front)
            igd[iterID,seed] = generational_distance(true_front, pareto_points)

            # Costs
            cost += np.asarray((experiment_log['cost']))[iterID]
            costs[iterID, seed] = cost

    return gd, igd, costs



def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    # True causal Pareto front 
    true_front = np.asarray(pd.read_csv(f'{problem_dir}/' + 'TrueCausalParetoFront.csv'))
    n_targets = true_front.shape[1]

    # Plotting
    plt.figure(figsize=(14, 6))

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    for exp_set in ['pomis', 'mobo']:
        gd, igd, costs = calculate_metrics(args, problem_dir, true_front, exp_set, n_targets)

        gd_avg, gd_std = np.mean(gd, axis=1), np.std(gd, axis=1)
        igd_avg, igd_std = np.mean(igd, axis=1), np.std(igd, axis=1)
        costs_avg = np.mean(costs, axis=1)

        print(f"GD: {gd_avg}, IGD: {igd_avg}, costs: {costs_avg}")

        # Legend labels mapping
        legend_labels = {
            'pomis': 'mo-cbo',
            'mobo': 'mobo',
            'mis': r'CPS + $\mathbb{M}_{\mathcal{G},\mathbf{Y}}$'
        }

        # Color mapping
        color_map = {
            'pomis': 'blue',
            'mobo': 'orange',
            'mis': 'black'
            }

        # Plot GD
        plt.plot(costs_avg, gd_avg, label=f'{legend_labels[exp_set]}', marker=None, linestyle='-', color=color_map[exp_set], linewidth=2)
        plt.fill_between(costs_avg, gd_avg - gd_std, gd_avg + gd_std, alpha=0.2, color=color_map[exp_set])

        # Plot IGD
        #plt.plot(costs_avg, igd_avg, label=f'{legend_labels[exp_set]}', marker=None, linestyle='-', color=color_map[exp_set], linewidth=3)
        #plt.fill_between(costs_avg, igd_avg - igd_std, igd_avg + igd_std, alpha=0.2, color=color_map[exp_set])
    
    # Health - IGD&GD
    #plt.xlim(0,110)
    #plt.xticks(fontname = "STIXGeneral")
    #plt.yticks(np.arange(0.0, 0.21, 0.1), fontname = "STIXGeneral")

    # Labels and title
    # mo-cbo1 - IGD
    #plt.ylim(0.5, 4.7)
    #plt.xlim(left=0)
    ## plt.yticks(np.arange(0, 5, 5))

    # mo-cbo1 - GD
    #plt.ylim(0.15, 0.9)
    #plt.xlim(0,110)
    #plt.yticks(np.arange(0.2, 1.0, 0.2)) 

    #plt.ylim(0,0.30)

    #plt.xlabel('cumulative intervention cost', fontsize=38)
    #plt.ylabel('IGD', fontsize=38) 
   
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.legend(fontsize=38) 
    plt.tick_params(axis='both', which='major', labelsize=38)
    plt.show()


if __name__ == '__main__':
    main()