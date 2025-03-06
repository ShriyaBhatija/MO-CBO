import argparse
import pathlib
from typing import OrderedDict 
import numpy as np
import pandas as pd

from problems import *
from helpers import *


parser = argparse.ArgumentParser(description='create_datasets')
parser.add_argument('--problem', default = 'mo-cbo1', type = str, help = 'problem name')
parser.add_argument('--exp-set', type=str, default='mobo', choices=['mo-cbo', 'mis', 'mobo'], help='exploration set')
parser.add_argument('--seed', default = 0, type = int, help = 'random seed')
parser.add_argument('--obs_num_samples', default=100, type=int, help='number of observational samples to be generated')
parser.add_argument('--int_num_samples', default=5, type=int, help='number of interventional samples to be generated')

def main(seed):
    args = parser.parse_args()

    np.random.seed(seed=seed)

    ## Set the parameters
    problem = args.problem
    obs_num_samples = args.obs_num_samples
    int_num_samples = args.int_num_samples


    # Create save folder if it doesn't exist
    pathlib.Path('Data/' + str(args.problem) + f'/{args.exp_set}/{seed}').mkdir(parents=True, exist_ok=True)

    if problem == 'mo-cbo1':
        observational_samples = OrderedDict([('X1', []), ('X2', []), ('X3', []), ('X4', []), ('Y1', []), ('Y2', []), ('control', [])])
        graph = MO_CBO1()
    elif problem == 'mo-cbo2':
        observational_samples = OrderedDict([('U', []), ('X1', []), ('X2', []), ('X3', []), ('X4', []), ('Y1', []), ('Y2', [])])
        graph = MO_CBO2()
    elif problem == 'mo-cbo-health':
        observational_samples = OrderedDict([('age', []), ('bmi', []), ('statin', []), ('aspirin', []), ('cancer', []), ('psa', []), ('control', [])])
        graph = Health()
    elif problem == 'mo-cbo3':
        observational_samples = OrderedDict([('U', []), ('X1', []), ('X2', []), ('X3', []), ('Y1', []), ('Y2', []), ('Y3', []), ('control', [])])
        graph = MO_CBO3()
    elif problem == 'mo-cbo-econ':
        graph = MO_CBO3()
        observational_samples = {key: [] for key, val in  graph.define_SEM().items()}
    


    targets = graph.get_targets()
    exploration_set = graph.get_exploration_sets()[args.exp_set]
        
    list_interventional_ranges = graph.get_interventional_ranges()

    
    observational_samples = pd.DataFrame([])
    for i in range(obs_num_samples):
        sample = sample_from_model(model = graph.define_SEM())
        observational_samples = pd.concat([observational_samples, pd.DataFrame([sample])], ignore_index=True)
        
    # Save as pkl file as in the folder
    # Save it using a protocol compatible with Python 3.7
    observational_samples.to_pickle('./Data/' + str(args.problem) + f'/{args.exp_set}/{seed}/' + 'observations.pkl', protocol=4)
    
    interventional_data = [] 

    for index, variables in enumerate(exploration_set):
        interventional_data.append([])
        interventional_data[index].append(len(variables))

        if len(variables) > 0:
            [interventional_data[index].append(var) for var in variables]

            interventions = [np.linspace(list_interventional_ranges[var][0], list_interventional_ranges[var][1], int_num_samples)[:, np.newaxis] for var in variables]
            interventions = np.concatenate(interventions, axis=1)

            # shuffle each column to get different combinations of interventions
            for i in range(interventions.shape[1]):
                np.random.shuffle(interventions[:, i])
            
            interventional_data[index].append(np.array(interventions))

            means = np.zeros((len(interventions), len(targets)))
            vars = np.zeros((len(interventions), len(targets)))

            for i in range(0, len(interventions)):
                print(i)
                intervention = [{var: interventions[i, j] for j, var in enumerate(variables)}]
                #new_model = intervene(*intervention, model=graph.define_SEM())
                for j, target in enumerate(targets):
                    means[i,j], vars[i,j] = compute_target_function(*intervention, model=graph.define_SEM(), target_variable=target, num_samples=1000)

            interventional_data[index].append(means)


        # Empty intervention set
        elif len(variables) == 0:
            [interventional_data[index].append(' ')]
            interventional_data[index].append([[]])
            
            means = np.zeros((int_num_samples, len(targets)))
            vars = np.zeros((int_num_samples, len(targets)))

            for i in range(0, int_num_samples):
                intervention = [{}]
                for j, target in enumerate(targets):
                    means[i,j], vars[i,j] = compute_target_function(*intervention, model=graph.define_SEM(), target_variable=target, num_samples=10000)

            interventional_data[index].append(means)

    interventional_data = np.array(interventional_data, dtype=object)
        
    # Save as npy file as in the folder
    np.save('./Data/' + str(args.problem) + f'/{args.exp_set}/{seed}/' + f'interventional_data.npy', interventional_data)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args.seed)