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

    pareto_points = []

    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'
        paretoEval_dict = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
        max_iterID = max(list(set(paretoEval_dict['iterID'])))

        # Get the points from the Pareto front of the last iteration (i.e. the complete approximation)
        points = paretoEval_dict[paretoEval_dict['iterID'] == max_iterID]
        for _, row in points.iterrows():
            pareto_points.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))

    
    # Extract x, y values and colors from the filtered points
    print(colours)
    pareto_f1_values = [point[1][0] for point in pareto_points]
    paretp_f2_values = [point[1][1] for point in pareto_points]
    colours = [colours[point[0]] for point in pareto_points]

    # Create a scatter plot with colors based on intervention sets
    plt.figure(figsize=(9, 5))
    plt.xlim(-3, 56) 
    plt.ylim(-2, 27)
    # Increase the thickness of the plot border (spines)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    plt.scatter(true_front['f1'], true_front['f2'], c='lightgray', s=90)
    plt.scatter(pareto_f1_values, paretp_f2_values, c=colours, s=90, edgecolors='black', linewidth=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


if __name__ == '__main__':
    main()