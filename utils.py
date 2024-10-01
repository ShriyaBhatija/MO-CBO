import os, sys
import yaml
import pandas as pd



def get_intervention_set_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        if str(intervention_variables[i]) != 'control':
            string += str(intervention_variables[i]) 
    return string


def get_result_dir(args):
    '''
    Get directory of result location (result/problem/algo/seed/)
    '''
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    exp_name = '' if args.exp_name is None else '-' + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, algo_name, args.mode, args.exp_set, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_experiment_log(args, experiment_log):
    '''
    Get directory of result location (result/problem/algo/seed/)
    '''
    experiment_log = pd.DataFrame(experiment_log)
    filepath = os.path.join(get_result_dir(args), 'experiment_log.csv')
    experiment_log.to_csv(filepath, index=False)


def save_args(general_args, framework_args):
    '''
    Save arguments to yaml file
    '''
    all_args = {'general': vars(general_args)}
    all_args.update(framework_args)

    result_dir = get_result_dir(general_args)
    args_path = os.path.join(result_dir, 'args.yml')

    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, 'w') as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)


def setup_logger(args):
    '''
    Log to file if needed
    '''
    logger = None

    if args.log_to_file:
        result_dir = get_result_dir(args)
        log_path = os.path.join(result_dir, 'log.txt')
        logger = open(log_path, 'w')
        sys.stdout = logger
    
    return logger
