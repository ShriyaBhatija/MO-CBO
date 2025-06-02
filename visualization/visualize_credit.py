# plot the causal Pareto front approximation as well as the ground truth causal Pareto front
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import get_problem_dir, get_intervention_sets, defaultColours
from metrics import is_pareto_optimal
from arguments import get_vis_args
from collections import OrderedDict


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict

class Credit():
    def __init__(self):
      super().__init__()
      self.Y = ['Y1', 'Y2']
      self.X = ['X3', 'X4', 'X6', 'X7']


    def define_SEM(self):
        def fX1(epsilon, **kwargs):
            return 0
        
        def fX2(epsilon, **kwargs):
            return 0
    
        def fX3(epsilon, X1, X2, **kwargs):
            return 0.5 + (1 + np.exp(-(-1+0.5*X1 + (1+np.exp(-0.1*X2)) ))) - 1
        
        def fX4(epsilon, X1, X2, **kwargs):
            return 1 + 0.01*(X2-5)*(5-X2) + X1 
        
        def fY1(epsilon, X1, X2, X4, **kwargs):
            return -1 + 0.01*X2 + 2*X1 + X4
        
        def fX6(epsilon, X1, X2, X3, **kwargs):
            return -4 + 0.1*(X2 + 35) + 2*X1 + X1*X3
        
        def fX7(epsilon, X6, **kwargs):
            return -4 + int(X6 > 0)
        
        def fY2(epsilon, X4, Y1, X6, X7, **kwargs):
            return 1./(1 + np.exp(-0.3*(-X4-Y1+X6+X7+X6*X7)))
    

        graph = OrderedDict([
            ('X1', fX1),
            ('X2', fX2),
            ('X3', fX3),
            ('X4', fX4),
            ('Y1', fY1),
            ('X6', fX6),
            ('X7', fX7),
            ('Y2', fY2)
        ])
        return graph


    def get_exploration_sets(self):
      mo_cbo = [['X4', 'X6', 'X7']]
      manipulative_variables = [self.X]

      exploration_sets = {
          'mo-cbo': mo_cbo,
          'mobo': manipulative_variables
      }
      return exploration_sets


    def get_interventional_ranges(self):
        dict_ranges = OrderedDict([
            ('X3', [-2, 2]),
            ('X4', [-2, 2]),
            ('X7', [-2, 2]),
            ('X6', [-2, 2]),
        ])
        return dict_ranges


def get_cbo_options():

    problems = {
        #'mo-cbo1': MO_CBO1(),
        #'mo-cbo2': MO_CBO2(),
        #'mo-cbo-health': Health(),
        #'mo-cbo-econ': Econ(),
        'mo-cbo-credit': Credit()
    }
    return problems

def sample_from_model(model, epsilon = None):
    if epsilon is None:
        epsilon = np.random.normal(0, 0.5, len(model))
    sample = {}
    for variable, function in model.items():
        sample[variable] = function(epsilon, **sample)
    return sample
  

def intervene(*interventions, model):
    new_model = model.copy()

    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions[0].items():
        assign(new_model, variable, value)
  
    return new_model


def compute_target_function(*interventions, model, target_variable, num_samples=100):
    mutilated_model = intervene(*interventions, model = model)

    samples = [sample_from_model(mutilated_model) for _ in range(num_samples)]
    samples = pd.DataFrame(samples)
    return np.mean(samples[target_variable]), np.var(samples[target_variable])


def intervene_dict(model, **interventions):

    new_model = model.copy()

    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions.items():
        assign(new_model, variable, value)
  
    return new_model


#EDITED to support our multi-objective optimisation framework
def Intervention_function(*interventions, model, targets, num_samples = 1000):

    def compute_target_function_fcn(value):
        num_interventions = len(interventions[0])

        for i in range(num_interventions):
            interventions[0][list(interventions[0].keys())[i]] = value[i]

        mutilated_model = intervene_dict(model, **interventions[0])
        
        samples = [sample_from_model(mutilated_model) for _ in range(num_samples)]
        samples = pd.DataFrame(samples)

        result = np.array([np.mean(samples[target]) for target in targets])
        return result

    return compute_target_function_fcn

def main():
    # get argument values and initializations
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    graph = get_cbo_options()[f'{args.problem}']
    target_function = Intervention_function(get_interventional_dict(['X4', 'X6', 'X7']),
									model = graph.define_SEM(), targets = graph.Y)

    directory_path = f'{problem_dir}/mo-cbo/dgemo/{args.seed}/'
    all_contents = os.listdir(directory_path)

    intervention_sets = [name for name in all_contents if os.path.isdir(os.path.join(directory_path, name))]
    colours = {}

    for i, intervention_set in enumerate(intervention_sets):
        colours[intervention_set] = defaultColours[i]

    # True causal Pareto front 
    #true_front = pd.read_csv(f'{problem_dir}/' + 'TrueCausalParetoFront.csv')
    #n_targets = len(true_front.columns)
    n_targets = 2

    all_pareto_points = []
    real = []
    # read result csvs 
    for intervention_set in intervention_sets:
        print(f'Processing intervention set: {intervention_set}')
        csv_folder = f'{directory_path}/{intervention_set}/'

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
                x_values = [row['x1'], row['x2'], row['x3']]
                y_pareto = target_function(x_values)
                real.append(y_pareto)
                #all_pareto_points.append((intervention_set, y_pareto))
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

    print(real)
    csv_folder =  f'{problem_dir}/mobo/{args.algo}/{args.seed}/X3X4X6X7/'
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
        #ax.scatter(true_front['f1'], true_front['f2'], true_front['f3'], c='hotpink', s=90,linewidth=0.6, alpha=0.35, depthshade=True, zorder=1)
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

        # Health example
        #plt.ylim(bottom=4.85, top=7.33)
        #plt.yticks(np.arange(5.0, 7.5, 0.5))

        #plt.scatter(true_front['f1'], true_front['f2'], c='lightgray', s=120)
        #plt.scatter(pareto_x_values, pareto_y_values, s=110,linewidth=0.6)
        plt.scatter(np.array(real)[:,0], np.array(real)[:,1], c='lightgray', s=120)
        #plt.scatter(pareto_x_values_mobo, pareto_y_values_mobo, s=110, linewidth=0.6)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.tight_layout(pad=1.0) 
        plt.show()



if __name__ == '__main__':
    main()