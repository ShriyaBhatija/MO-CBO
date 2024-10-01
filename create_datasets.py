import argparse
import pathlib 
import numpy as np
import pandas as pd

from problems import *
from utils_functions import *


parser = argparse.ArgumentParser(description='create_datasets')
parser.add_argument('--experiment', default = 'cbo1', type = str, help = 'experiment')
parser.add_argument('--exp-set', type=str, default='mis', choices=['mis', 'pomis', 'mobo'], help='exploration set')
parser.add_argument('--obs_num_samples', default=1000, type=int, help='number of observational samples to be generated')
parser.add_argument('--int_num_samples', default=20, type=int, help='number of interventional samples to be generated')

def main(seed):
    args = parser.parse_args()

    np.random.seed(seed=seed)


    ## Set the parameters
    experiment = args.experiment
    obs_num_samples = args.obs_num_samples
    int_num_samples = args.int_num_samples


    # Create save folder if it doesn't exist
    pathlib.Path('Data/' + str(args.experiment) + f'/{seed}').mkdir(parents=True, exist_ok=True)


    if experiment == 'cbo1':
        observational_samples = OrderedDict([('X1', []), ('X2', []), ('Z1', []), ('Z2', []), ('Y1', []), ('Y2', []), ('control', [])])
        graph = CBO1(observational_samples)

    targets = graph.get_targets()
    exploration_set = graph.get_exploration_sets()[args.exp_set]
        
    list_interventional_ranges = graph.get_interventional_ranges()


    observational_samples = pd.DataFrame([])
    for i in range(obs_num_samples):
        sample = sample_from_model(model = graph.define_SEM())
        observational_samples = pd.concat([observational_samples, pd.DataFrame([sample])], ignore_index=True)
        
    # Save as pkl file as in the folder
    # Save it using a protocol compatible with Python 3.7
    observational_samples.to_pickle('./Data/' + str(args.experiment) + f'/{seed}/' + 'observations.pkl', protocol=4)


    interventional_data = [] 

    for index, variables in enumerate(exploration_set):
        interventional_data.append([])
        interventional_data[index].append(len(variables))
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
            intervention = [{var: interventions[i, j] for j, var in enumerate(variables)}]
            #new_model = intervene(*intervention, model=graph.define_SEM())
            for j, target in enumerate(targets):
                means[i,j], vars[i,j] = compute_target_function(*intervention, model=graph.define_SEM(), target_variable=target, num_samples=1000)

        interventional_data[index].append(means)
            
    interventional_data = np.array(interventional_data, dtype=object)
        
    # Save as npy file as in the folder
    np.save('./Data/' + str(args.experiment) + f'/{seed}/' + f'interventional_data.npy', interventional_data)



if __name__ == '__main__':
    for seed in range(10):
        main(seed)