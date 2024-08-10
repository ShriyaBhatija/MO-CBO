import os, sys
import yaml

from visualization.utils import get_result_dir


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
