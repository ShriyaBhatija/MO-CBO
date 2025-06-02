from argparse import ArgumentParser

'''
Arguments for visualisation
'''

def get_vis_args():
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default = 'mo-cbo-credit',
        help='optimization problem')
    parser.add_argument('--exp-set', type=str, default = 'mo-cbo', choices=['mis', 'mo-cbo', 'mobo'],
        help='exploration set')
    parser.add_argument('--mode', type=str, default='int_data', choices=['causal_prior', 'int_data'], 
        help='which samples to do the initial iteration with')
    parser.add_argument('--seed', type=int, default=1,
        help='seed / test run')
    parser.add_argument('--metric', type=str, default = 'igd', choices=['gd', 'igd'],
        help='performance metric')
    
    parser.add_argument('--algo', type=str, default='dgemo',
        choices=['dgemo', 'parego', 'moead-ego', 'tsemo'],
        help='type of algorithm to use with some predefined arguments, or custom arguments')


    args = parser.parse_args()
    return args