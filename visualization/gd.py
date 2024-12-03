# plot the causal Pareto front approximation as well as the ground truth causal Pareto front
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_problem_dir, get_intervention_sets, defaultColours
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

    # Compute the average and apply the final root
    return (np.sum(distances) / len(A)) ** (1 / p)




def main():

    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)
    intervention_sets = get_intervention_sets(args)

    # True causal Pareto front 
    true_front = np.asarray(pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/' + 'TrueCausalParetoFront.csv'))

    cost = len(np.asarray(pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/' + 'experiment_log.csv')['cost']))
    print(cost)

    
    all_pareto_points = []

    gd_mean_iterations = []
    gd_var_iterations = []
    gd_std_iterations = []

    n_iterations = 19

    for iteration in range(0, n_iterations):

        gd_seeds = []

        for seed in range(0,9):
            args.seed = seed
            for intervention_set in intervention_sets:
                pareto_points = np.array((len(intervention_sets), 2))

                csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'
                paretoEval = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
                max_iterID = max(list(set(paretoEval['iterID'])))
                
                if max_iterID >= iteration:
                    iterID = iteration
                else:
                    iterID = max_iterID

                # Get the points from the Pareto front of the chosen iteration (i.e. the complete approximation)
                points = paretoEval[paretoEval['iterID'] == iterID]
                for _, row in points.iterrows():
                    all_pareto_points.append([row['Pareto_f1'], row['Pareto_f2']])

            # Calculate Pareto efficient points
            pareto_flags = is_pareto_optimal(np.array([point for point in all_pareto_points]))
            pareto_points = np.asarray([value for value, flag in zip(all_pareto_points, pareto_flags) if flag])
        
            gd_seeds.append(generational_distance(true_front, pareto_points))
    
        gd_mean_iterations.append(np.mean(gd_seeds))
        gd_var_iterations.append(np.var(gd_seeds))
        gd_std_iterations.append(np.sqrt(gd_var_iterations))




if __name__ == '__main__':
    main()


