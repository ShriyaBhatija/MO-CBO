# plot the causal Pareto front approximation as well as the ground truth causal Pareto front

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_problem_dir, get_intervention_sets, defaultColours
from metrics import is_pareto_optimal
from arguments import get_vis_args


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    intervention_sets = get_intervention_sets(args)
    colours = {}
    for i, intervention_set in enumerate(intervention_sets):
        colours[intervention_set] = defaultColours[i]

    # True causal Pareto front 
    true_front = pd.read_csv(f'{problem_dir}/' + 'TrueCausalParetoFront.csv')
    n_targets = len(true_front.columns)

    all_pareto_points = []

    # read result csvs 
    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'
        print(csv_folder)

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


    
    csv_folder = f'/Users/shriyabhatija/Desktop/Causal_ParetoSelect/result/mo-cbo-health/int_data/mobo/{args.seed}/ciweightbmiaspirin/'
    paretoEval = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
    max_iterID = max(list(set(paretoEval['iterID'])))
    points = paretoEval[paretoEval['iterID'] == max_iterID]
    pareto_points_mobo = []
    for _, row in points.iterrows():
        pareto_points_mobo.append((intervention_set, [row['Pareto_f1'], row['Pareto_f2']]))
    pareto_x_values_mobo = [point[1][0] for point in pareto_points_mobo]
    pareto_y_values_mobo = [point[1][1] for point in pareto_points_mobo]


    if n_targets == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$Y_1$', fontsize=24, labelpad=10)
        ax.set_ylabel(r'$Y_2$', fontsize=24, labelpad=14)
        ax.set_zlabel(r'$Y_3$', fontsize=24, labelpad=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.scatter(true_front['f1'], true_front['f2'], true_front['f3'], c='hotpink', s=90,linewidth=0.6, alpha=0.35, depthshade=True, zorder=1)
        ax.scatter(pareto_x_values, pareto_y_values, pareto_z_values, c=colours, s=90, edgecolors='black', linewidth=0.6, depthshade=False, zorder=2)

        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()

    if n_targets == 2:
        plt.figure(figsize=(9, 5))
        #plt.xlabel(r'$Y_1$', fontsize=24, labelpad=10) 
        #plt.ylabel(r'$Y_2$', fontsize=24, labelpad=14) 
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.8)
        plt.scatter(true_front['f1'], true_front['f2'], c='lightgray', s=120)
        plt.scatter(pareto_x_values, pareto_y_values, s=110,linewidth=0.6)
        plt.scatter(pareto_x_values_mobo, pareto_y_values_mobo, s=110, linewidth=0.6)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.tight_layout() 
        plt.show()



if __name__ == '__main__':
    main()