import numpy as np

from helpers import *
from problems.common import get_cbo_options

from arguments import get_args
from C_ParetoSelect import Causal_ParetoSelect


def main(args, framework_args):
    # set global seed
    np.random.seed(args.seed)

    # get the graph
    graph = get_cbo_options()[f'{args.problem}']

    # define the exploration set
    exploration_set = graph.get_exploration_sets()[args.exp_set]

    # define cost structure
    costs = graph.get_cost_structure(type_cost = args.type_cost)

    # interventional data
    interventional_data = np.load(f'./Data/{args.problem}/{args.exp_set}/{args.seed}/interventional_data.npy', allow_pickle=True)
    
    # run the algorithm
    Causal_ParetoSelect(args,framework_args, graph, exploration_set, costs, interventional_data)


if __name__ == '__main__': 
    for seed in range(8,10):
        args, framework_args = get_args()
        args.seed = seed
        main(args, framework_args)