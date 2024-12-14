# plot the causal Pareto front approximation as well as the ground truth causal Pareto front

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from utils import get_problem_dir, get_intervention_sets, defaultColours
from arguments import get_vis_args

import numpy as np


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


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    intervention_sets = get_intervention_sets(args)
    colours = {}
    for i, intervention_set in enumerate(intervention_sets):
        colours[intervention_set] = defaultColours[i]

    # True causal Pareto front 
    true_front = pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/' + 'TrueCausalParetoFront.csv')
    n_targets = len(true_front.columns)

    all_pareto_points = []

    # read result csvs 
    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'

        if intervention_set == 'empty':
            points = pd.read_csv(csv_folder + 'sample.csv')
            for _, row in points.iterrows():
                all_pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))
            continue
    
        paretoEval = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
        max_iterID = max(list(set(paretoEval['iterID'])))

        # Get the points from the Pareto front of the last iteration (i.e. the complete approximation)
        points = paretoEval[paretoEval['iterID'] == max_iterID]
        for _, row in points.iterrows():
            if n_targets == 2:
                all_pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))
            elif n_targets == 3:
                all_pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2'], row['Pareto_f3']]))

    # Calculate Pareto efficient points
    pareto_flags = is_pareto_optimal(np.array([point[1] for point in all_pareto_points]))
    pareto_points = [(intervention_set, value) for (intervention_set, value), flag in zip(all_pareto_points, pareto_flags) if flag]
    # Extract x, y values and colors from the filtered points
    pareto_x_values = [point[1][0] for point in pareto_points]
    pareto_y_values = [point[1][1] for point in pareto_points]
    if n_targets == 3:
        pareto_z_values = [point[1][2] for point in pareto_points]
    colours = [colours[point[0]] for point in pareto_points]


    if n_targets == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$Y_1$', fontsize=24, labelpad=10)
        ax.set_ylabel(r'$Y_2$', fontsize=24, labelpad=14)
        ax.set_zlabel(r'$Y_3$', fontsize=24, labelpad=14)
        # Increase the thickness of the plot border (spines)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

        # Plot the points
        ax.scatter(true_front['f1'], true_front['f2'], true_front['f3'], c='hotpink', s=90,linewidth=0.6, alpha=0.35, depthshade=True, zorder=1)
        ax.scatter(pareto_x_values, pareto_y_values, pareto_z_values, c=colours, s=90, edgecolors='black', linewidth=0.6, depthshade=False, zorder=2)

        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()

    if n_targets == 2:
        # Create a scatter plot with colors based on intervention sets
        plt.figure(figsize=(9, 5))
        plt.xlabel(r'$Y_1$', fontsize=24, labelpad=10) 
        plt.ylabel(r'$Y_2$', fontsize=24, labelpad=14) 
        # Increase the thickness of the plot border (spines)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.8)
        plt.scatter(true_front['f1'], true_front['f2'], c='lightgray', s=90)
        plt.scatter(pareto_x_values, pareto_y_values, c=colours, s=90, edgecolors='black', linewidth=0.6)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout() # Adjust layout to ensure everything fits
        plt.show()



if __name__ == '__main__':
    main()