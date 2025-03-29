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
    completed = 0 
    for seed in range(0,40):
        for approach in ["mo-cbo", 'mobo']:
            try:
                args, framework_args = get_args()
                args.seed = seed
                args.exp_set = approach
                main(args, framework_args)
                completed+=1
                if completed == 8:
                    break
                with open("runs.log", "a") as f:
                    f.write(f"Succeeded for seed {seed} and approch {approach}.")
            
            except Exception as e:
                with open("runs.log", "a") as f:
                    f.write(f"Failed for seed {seed} and approch {approach}. Due to error {str(e)}. Skipping...")
                    continue