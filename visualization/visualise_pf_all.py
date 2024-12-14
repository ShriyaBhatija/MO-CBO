# plot the approximated Pareto fronts for each intervention set

import os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arguments import get_vis_args
from utils import get_problem_dir, get_intervention_sets, defaultColours

def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    intervention_sets = get_intervention_sets(args)
    colours = {}
    for i, intervention_set in enumerate(intervention_sets):
        colours[intervention_set] = defaultColours[i]

    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # true causal Pareto front 
    true_front = pd.read_csv(f'{problem_dir}/{args.algo}/{args.mode}/' + 'TrueCausalParetoFront.csv')
    n_targets = len(true_front.columns)

    pareto_points = []

    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'

        if intervention_set == 'empty':
            points = pd.read_csv(csv_folder + 'sample.csv')
            for _, row in points.iterrows():
                pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))
            continue

        paretoEval_dict = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
        max_iterID = max(list(set(paretoEval_dict['iterID'])))

        # Get the points from the Pareto front of the last iteration (i.e. the complete approximation)
        points = paretoEval_dict[paretoEval_dict['iterID'] == max_iterID]
        for _, row in points.iterrows():
            if n_targets == 2:
                pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))
            elif n_targets == 3:
                pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2'], row['Pareto_f3']]))

    
    # Extract x, y values and colors from the filtered points
    print(colours)
    pareto_f1_values = [point[1][0] for point in pareto_points]
    paretp_f2_values = [point[1][1] for point in pareto_points]
    if n_targets == 3:
        paretp_f3_values= [point[1][2] for point in pareto_points]
    colours = [colours[point[0]] for point in pareto_points]


    if n_targets == 2:
        # Create a scatter plot with colors based on intervention sets
        plt.figure(figsize=(9, 5))
        #plt.xlim(-3, 52) 
        #plt.ylim(-2, 26)
        plt.xlabel(r'$Y_1$', fontsize=24, labelpad=10) 
        plt.ylabel(r'$Y_2$', fontsize=24, labelpad=14) 
        # Increase the thickness of the plot border (spines)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.8)
        plt.scatter(true_front['f1'], true_front['f2'], c='lightgray', s=90)
        plt.scatter(pareto_f1_values, paretp_f2_values, c=colours, s=90, edgecolors='black', linewidth=0.6)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout() # Adjust layout to ensure everything fits
        plt.show()


    elif n_targets == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(r'$Y_1$', fontsize=24, labelpad=10)
        ax.set_ylabel(r'$Y_2$', fontsize=24, labelpad=14)
        ax.set_zlabel(r'$Y_3$', fontsize=24, labelpad=14)

        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

        ax.scatter(true_front['f1'], true_front['f2'], true_front['f3'], c='hotpink', s=90,linewidth=0.6, alpha=0.35, depthshade=True, zorder=2)
        ax.scatter(pareto_f1_values, paretp_f2_values, paretp_f3_values, c=colours, s=90, edgecolors='black', linewidth=0.6, depthshade=False, zorder=1)

        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()