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
    paretoEval_dict = {}
    max_iterIDs = {}
    for intervention_set in intervention_sets:
        csv_folder = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/{intervention_set}/'
        paretoEval_dict[intervention_set] = pd.read_csv(csv_folder + 'ParetoFrontEvaluated.csv')
        max_iterIDs[intervention_set] = max(list(set(paretoEval_dict[intervention_set]['iterID'])))


    # calculate proper range of plot
    minX = min([min(data['Pareto_f1']) for data in paretoEval_dict.values()])
    maxX = max([max(data['Pareto_f1']) for data in paretoEval_dict.values()])
    minY = min([min(data['Pareto_f2']) for data in paretoEval_dict.values()])
    maxY = max([max(data['Pareto_f2']) for data in paretoEval_dict.values()])

    plot_range_x = [minX - (maxX - minX), maxX + 0.05 * (maxX - minX)]
    plot_range_y = [minY - (maxY - minY), maxY + 0.05 * (maxY - minY)]

    colours = [
        'rgb(31, 119, 180)',  # Blue
        'rgb(214, 39, 40)',   # Red
        'rgb(255, 255, 0)',  # Yellow
        'rgb(255, 127, 14)',  # Orange
        'rgb(128, 0, 128)', # Purple
        'rgb(44, 160, 44)',   # Green
    ]

    # starting the figure
    fig = go.Figure()

    # Holds the min and max traces for each step
    stepTrace = []

    # Iterating through all the Potential Steps 
    for step in range(max(max_iterIDs.values())+1):
        traceStart = len(fig.data)
        scatter = go.Scatter

        for s, intervention_set in enumerate(intervention_sets):
            iterID = max_iterIDs[intervention_set] if step > max_iterIDs[intervention_set] else step
            paretoEval_trimmed = paretoEval_dict[intervention_set][paretoEval_dict[intervention_set]['iterID'] == iterID]

            # Sort points based on x-coordinates (or y-coordinates if the curve is vertical)
            sorted_points = paretoEval_trimmed.sort_values(by='Pareto_f1')

            # Evaluated Pareto front points
            trace_dict = dict(
                name = f'{intervention_set}',
                visible=False,
                mode='markers', 
                x=sorted_points['Pareto_f1'].values, 
                y=sorted_points['Pareto_f2'].values, 
                #hovertext=paretoEval_trimmed['hovertext'],
                hoverinfo="text",
                marker=dict(
                    color=colours[s],
                    symbol = 'circle',
                    size=10,
                    line=dict(
                        color='rgb(0, 0, 0)',
                        width=0.5
                    )
                ),
                line=dict(
                    color=f'rgba({colours[s][4:-1]}, 0.5)',  # Assuming colours[s] is in 'rgb(r, g, b)' format
                    width=5  # Adjust the width to make the lines thicker
                )
            )
            fig.add_trace(scatter(**trace_dict))
            
                
        traceEnd = len(fig.data)-1
        stepTrace.append([i for i in range(traceStart,traceEnd+1)])


    # Make Last trace visible
    for i in stepTrace[-1]:
        fig.data[i].visible = True
        scene_dict = dict(xaxis=dict(range=plot_range_x), yaxis=dict(range=plot_range_y))
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
        xaxis = dict(
            title = 'f1',
            range=plot_range_x,
            tickfont=dict(size=200),
        ),
        yaxis = dict(range=plot_range_y),
    )


    fig.update_layout(
        sliders=sliders,
        width=1300,
        height=900,
        title=f"Performance Space of {problem_name} using {args.algo}",
        xaxis = dict(
            range=plot_range_x,
            tickfont=dict(size=22)
        ),
        yaxis = dict(
            range=plot_range_y,
            tickfont=dict(size=24)
        ),
        autosize = False,
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            font=dict(
                size=14  # Change the font size here
            )
        )    
    )
        
    fig.show()


if __name__ == '__main__':
    main()