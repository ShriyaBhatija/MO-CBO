# plot each iteration of predicted Pareto front, proposed points, evaluated points for two algorithms

import os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from arguments import get_vis_args
from utils import get_problem_dir, get_intervention_sets


def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    intervention_sets = get_intervention_sets(args)

    problem_name = os.path.basename(os.path.dirname(problem_dir))

    # read result csvs 
    # paretoEval_list = {intervention_set1: paretoEval_list,..., intervention_setN: paretoEval_list}
    paretoEval_list = {}
    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'
        paretoEval_list[intervention_set] = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')

    true_front_file = os.path.join(problem_dir, 'TrueParetoFront.csv')
    has_true_front = os.path.exists(true_front_file)
    if has_true_front:
        df_truefront = pd.read_csv(true_front_file)

    n_obj = len([key for key in paretoEval_list[intervention_sets[0]] if key.startswith('Pareto_f')])

    # calculate proper range of plot
    minX = min([min(data['Pareto_f1']) for data in paretoEval_list.values()])
    maxX = max([max(data['Pareto_f1']) for data in paretoEval_list.values()])
    minY = min([min(data['Pareto_f2']) for data in paretoEval_list.values()])
    maxY = max([max(data['Pareto_f2']) for data in paretoEval_list.values()])
    if has_true_front:
        minX = min(min(df_truefront['f1']), minX)
        maxX = max(max(df_truefront['f1']), maxX)
        minY = min(min(df_truefront['f2']), minY)
        maxY = max(max(df_truefront['f2']), maxY)
    plot_range_x = [minX - (maxX - minX), maxX + 0.05 * (maxX - minX)]
    plot_range_y = [minY - (maxY - minY), maxY + 0.05 * (maxY - minY)]
    if n_obj > 2:
        minZ = min([min(data['Pareto_f3']) for data in paretoEval_list.values()])
        maxZ = max([max(data['Pareto_f3']) for data in paretoEval_list.values()])
        if has_true_front:
            minZ = min(min(df_truefront['f3']), minZ)
            maxZ = max(max(df_truefront['f3']), maxZ)
        plot_range_z = [minZ - (maxZ - minZ), maxZ + 0.05 * (maxZ - minZ)]

    # starting the figure
    fig = go.Figure()


    # Holds the min and max traces for each step
    stepTrace = []

    # Iterating through all the Potential Steps 
    for step in list(set(paretoEval_list[intervention_sets[1]]['iterID'])):
        traceStart = len(fig.data)
        scatter = go.Scatter if n_obj == 2 else go.Scatter3d

        for intervention_set in intervention_sets: 
            paretoEval_trimmed = paretoEval_list[intervention_set][paretoEval_list[intervention_set]['iterID'] == step]

            if intervention_set == intervention_sets[0]:
                colour = 'blue'
            elif intervention_set == intervention_sets[1]:
                colour = 'red'
            elif intervention_set == intervention_sets[2]:
                colour = 'green'
            elif intervention_set == intervention_sets[3]:
                colour = 'purple'

            # Evaluated Pareto front points
            trace_dict = dict(
                name = f'Pareto Front Evaluated for {intervention_set}',
                visible=False,
                mode='markers', 
                x=paretoEval_trimmed['Pareto_f1'], 
                y=paretoEval_trimmed['Pareto_f2'], 
                #hovertext=paretoEval_trimmed['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color=colour,
                    symbol = 'circle',
                    size=8 if n_obj == 2 else 4,
                    line=dict(
                        color='rgb(0, 0, 0)',
                        width=0.5
                    )
                )
            )
            if n_obj > 2: trace_dict['z'] = paretoEval_trimmed['Pareto_f3']
            fig.add_trace(scatter(**trace_dict))
            
            
        # Adding true Pareto front points
        if has_true_front:
            trace_dict = dict(
                name = 'True Pareto Front',
                visible=False,
                mode='markers', 
                x=df_truefront['f1'], 
                y=df_truefront['f2'], 
                marker=dict(
                    color='rgba(105, 105, 105, 0.8)',
                    size=2,
                    symbol='circle',
                )
            )
            if n_obj > 2: trace_dict['z'] = df_truefront['f3']
            fig.add_trace(scatter(**trace_dict))
                
        traceEnd = len(fig.data)-1
        stepTrace.append([i for i in range(traceStart,traceEnd+1)])


    # Make Last trace visible
    for i in stepTrace[-1]:
        fig.data[i].visible = True
        scene_dict = dict(xaxis=dict(range=plot_range_x), yaxis=dict(range=plot_range_y))
        if n_obj > 2:
            scene_dict['zaxis'] = dict(range=plot_range_z)
        fig.update_layout(scene=scene_dict)

    # Create and add slider
    steps = []
    j = 1
    for stepIndexes in stepTrace:
        # Default set everything Invisivible
        iteration = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=str(j-1)
        )
        j = j + 1
        #Toggle Traces in this Step to Visible
        for i in stepIndexes:
            iteration['args'][1][i] = True
        steps.append(iteration)

    sliders = [dict(
        active=int(len(steps))-1,
        currentvalue={"prefix": "Iteration: "},
        pad={"t": 50},
        steps=steps
    )]

    # Adding some Formatting to the Plot
    scene_dict = dict(
        xaxis_title='f1',
        yaxis_title='f2',
        xaxis = dict(range=plot_range_x),
        yaxis = dict(range=plot_range_y),
    )
    if n_obj > 2:
        scene_dict['zaxis_title'] = 'f3'
        scene_dict['zaxis'] = dict(range=plot_range_z)

    fig.update_layout(
        sliders=sliders,
        width=1300,
        height=900,
        title=f"Performance Space of {problem_name} using {args.algo}",
        scene = scene_dict,
        autosize = False
    )
        
    fig.show()


if __name__ == '__main__':
    main()