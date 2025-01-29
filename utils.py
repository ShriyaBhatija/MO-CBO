import os, sys
import yaml
import pandas as pd



def get_intervention_set_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        if str(intervention_variables[i]) != 'control':
            string += str(intervention_variables[i]) 
    return string



def calculate_batch_cost(intervention_set, costs, intervention_batch):
    '''
    Calculate cost for a batch of interventions
    '''
    current_cost = 0
    for i in range(len(intervention_batch)):
        for s, var in enumerate(intervention_set):
            current_cost += costs[var](intervention_batch[i][s])

    return current_cost


def get_problem_dir(args):
    '''
    Get directory of problem location (result/problem/)
    '''
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    result_dir = os.path.join(top_dir, args.problem)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def get_result_dir(args):
    '''
    Get directory of result location (result/problem/algo/seed/)
    '''
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    result_dir = os.path.join(top_dir, args.problem, args.mode, args.exp_set, str(args.seed))
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


def write_truefront_csv(args, truefront):
    '''
    Export true pareto front to a csv file
    '''
    problem_dir = get_problem_dir(args)
    filepath = os.path.join(problem_dir, 'TrueCausalParetoFront.csv')

    d = {}
    for i in range(truefront.shape[1]):
        col_name = f'f{i + 1}'
        d[col_name] = truefront[:, i]

    export_tf = pd.DataFrame(data=d)
    export_tf.to_csv(filepath, index=False)
