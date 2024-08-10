from argparse import ArgumentParser

'''
Arguments for visualization
'''

def get_vis_args():
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default = 'cbo1',
        help='optimization problem')
    parser.add_argument('--mis', type=str, default = 'X1X2',
        help='minimal intervention set')
    parser.add_argument('--mode', type=str, default='causal_prior', choices=['causal_prior', 'int_data', 'optimal'], 
        help='which samples to do the initial iteration with')
    parser.add_argument('--algo', type=str, nargs='+', default=None, 
        help='algorithm name / list of algorithm names for comparison, include exp-name if it is also specified, \
            None means compare all algorithms found')
    parser.add_argument('--family', default=True, action='store_true',
        help='whether has family / pareto front approximation')
    parser.add_argument('--seed', type=int, default=0,
        help='seed / test run')
    parser.add_argument('--n-seed', type=int, default=1,
        help='number of total seeds / test runs')

    parser.add_argument('--subfolder', type=str, default='default',
        help='subfolder name for storing results')
    parser.add_argument('--savefig', default=False, action='store_true',
        help='saving as png instead of showing the plot')

    args = parser.parse_args()
    return args
