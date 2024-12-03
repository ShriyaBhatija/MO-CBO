import os 
os.environ['OMP_NUM_THREADS'] = '1' # speed up

import numpy as np
import pandas as pd

from utils_functions import *
from problems.common import get_cbo_options

from arguments import get_args
from C_ParetoSelect import Causal_ParetoSelect


def main(args, framework_args):

    # set seed
    seed = args.seed
    np.random.seed(args.seed)

    ## Import the data
    full_observational_samples = pd.read_pickle(f'./Data/{args.problem}/{args.exp_set}/{args.seed}/observations.pkl')
    observational_samples = full_observational_samples[:args.n_init_sample_obs]

    ## Get the graph
    graph = get_cbo_options(observational_samples)[f'{args.problem}']

    ## Define the exploration set
    exploration_set = graph.get_exploration_sets()[args.exp_set]

    ## Define cost structure
    costs = graph.get_cost_structure(type_cost = args.type_cost)

    ## Givent the data fit all models used for do calculus
    functions = graph.fit_all_models()


    ## Define optimisation sets and the set of manipulative variables
    max_N = args.n_init_sample_obs + 50
    manipulative_variables = graph.get_set_MOBO()


    ## Define interventional ranges for all interventional variables and create a dict to store them
    dict_ranges = graph.get_interventional_ranges()

    ## Compute observation coverage
    #alpha_coverage, hull_obs, coverage_total = compute_coverage(observational_samples, manipulative_variables, dict_ranges)

    # Interventional data
    interventional_data = np.load(f'./Data/{args.problem}/{args.exp_set}/{args.seed}/interventional_data.npy', allow_pickle=True)

    # Initialise data lists
    data_x_list, data_y_list = define_initial_data_CBO(interventional_data, args.batch_size, exploration_set, seed)

    # Run the algorithm
    Causal_ParetoSelect(args, framework_args, graph, exploration_set, costs, functions, full_observational_samples, observational_samples, interventional_data, dict_ranges, data_x_list, data_y_list)



if __name__ == '__main__': 
    # load arguments
    args, framework_args = get_args()

    for seed in range(0,1):
        args.seed = seed
        main(args, framework_args)