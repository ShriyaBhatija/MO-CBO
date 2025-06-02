import pandas as pd
import numpy as np

from utils import get_problem_dir
from arguments import get_vis_args

def calculate_runtime(args, problem_dir, exp_set, algo):
    args.exp_set = exp_set
    args.algo = algo

    avg_runtime = []

    for seed in range(0,10):
        args.seed = seed
        experiment_log = pd.read_csv(f'{problem_dir}/{args.exp_set}/{args.algo}/{args.seed}/' + 'experiment_log.csv')[1:]
        time = experiment_log['time'].to_numpy()
        total_time = time.mean()
        avg_runtime.append(total_time)
    
    return np.array(avg_runtime).mean()


if __name__ == '__main__':
    args = get_vis_args()
    problem_dir = get_problem_dir(args)

    mean = calculate_runtime(args, problem_dir, 'mo-cbo', 'dgemo')
    print(mean)