# plot the causal Pareto front approximation as well as the ground truth causal Pareto front
import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_problem_dir, get_intervention_sets, get_intervention_set_name
from arguments import get_vis_args


def is_pareto_optimal(point_cloud):
    """
    Find the Pareto-optimal points among a set of points.
    :param points: An (n_points, n_point_cloud) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto optimal
    """
    is_optimal= np.ones(point_cloud.shape[0], dtype = bool)
    for i, c in enumerate(point_cloud):
        if is_optimal[i]:
            is_optimal[is_optimal] = np.any(point_cloud[is_optimal]<c, axis=1)  
            is_optimal[i] = True  
    return is_optimal



def generational_distance(A, Z, p=2):
    """
    Compute the Generational Distance (GD) between a set of solutions and a reference Pareto-front.

    Parameters:
    - A (ndarray): Array of shape (m, n), where m is the number of solutions, and n is the number of objectives.
    - Z (ndarray): Array of shape (k, n), where k is the number of reference points, and n is the number of objectives.
    - p (int): The norm to use for calculating distances (default is p=2 for L2 norm).

    Returns:
    - gd (float): The Generational Distance value.
    """

    # Ensure inputs are numpy arrays
    A = np.array(A)
    Z = np.array(Z)

    # Compute the distance of each point in A to the closest point in Z
    distances = []
    for a in A:
        # Compute the distance from point `a` to all points in `Z` and take the minimum
        min_distance = np.min(np.linalg.norm(Z - a, ord=p, axis=1))
        distances.append(min_distance ** p)  # Raise to power `p`

        #print(f"a: {a}, z: {Z[np.argmin(np.linalg.norm(Z - a, ord=p, axis=1))]}, dist: {min_distance}")

    # Compute the average and apply the final root
    return ((np.sum(distances)) / len(A))**(1/2)


def calculate_metrics(args, problem_dir, true_front, exp_set, n_targets):
    args.exp_set = exp_set

    # Determine the minimum number of iterations across all seeds
    min_n_iter = np.inf

    for seed in range(0, 10):
        args.seed = seed
        experiment_log = pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/' + 'experiment_log.csv')[1:]
        min_n_iter = min(min_n_iter, len(experiment_log))

    # Initialize arrays with the maximum number of iterations
    gd = np.zeros((min_n_iter+1, 10))
    igd = np.zeros((min_n_iter+1, 10))
    costs = np.zeros((min_n_iter+1, 10))

    for seed in range(0,10):
        args.seed = seed
        cost = 0

        count: Dict[str, int] = {}
        all_pareto_points = []

        experiment_log = pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/' + 'experiment_log.csv')

        for iterID in range(0, min_n_iter+1):

            if iterID == 0:
                 # Before any iteration, go over all sets and get the initial Pareto points
                intervention_sets = set(np.asarray((experiment_log['intervened_set']))[1:])
                for i_set in intervention_sets:
                    csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{i_set}/'
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
                    csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{i_set}/'
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
    true_front = np.asarray(pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/' + 'TrueCausalParetoFront.csv'))
    n_targets = true_front.shape[1]

    # Plotting
    plt.figure(figsize=(10, 6))

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
            'pomis': r'CPS + $\mathbb{P}_{\mathcal{G},\mathbf{Y}}$',
            'mobo': 'baseline',
            'mis': r'CPS + $\mathbb{M}_{\mathcal{G},\mathbf{Y}}$'
        }

        # Color mapping
        color_map = {
            'pomis': 'blue',
            'mobo': 'green',
            'mis': 'black'
            }

        # Plot GD
        #plt.plot(costs_avg, gd_avg, label=f'{legend_labels[exp_set]}', marker=None, linestyle='-', color=color_map[exp_set], linewidth=2)
        #plt.fill_between(costs_avg, gd_avg - gd_std*0.6, gd_avg + gd_std*0.6, alpha=0.2, color=color_map[exp_set])

        # Plot IGD
        plt.plot(costs_avg, igd_avg, label=f'{legend_labels[exp_set]}', marker=None, linestyle='-', color=color_map[exp_set], linewidth=2)
        plt.fill_between(costs_avg, igd_avg - igd_std*0.6, igd_avg + igd_std*0.6, alpha=0.2, color=color_map[exp_set])
        
    
    # Labels and title
    # mo-cbo1 - IGD
    #plt.ylim(0.5, 4.7)
    #plt.xlim(left=0)
    ## plt.yticks(np.arange(0, 5, 5))

    # mo-cbo1 - GD
    #plt.ylim(0.15, 0.9)
    plt.xlim(left=0)
    #plt.yticks(np.arange(0.2, 1.0, 0.2)) 

    plt.xlabel('cost', fontsize=20)
    plt.ylabel('inverted generational distance', fontsize=20) 
    plt.legend(fontsize=20) 
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


if __name__ == '__main__':
    main()