import os


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
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    exp_name = '' if args.exp_name is None else '-' + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, algo_name, args.mode, args.exp_set, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def get_problem_dir(args):
    '''
    Get problem directory under result directory
    '''
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    problem_dir = os.path.join(result_dir, args.problem)
    return problem_dir

def get_intervention_sets(args):
    '''
    Get names of all intervention sets under the specified directory
    '''
    
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    problem_dir = os.path.join(result_dir, args.problem)

    directory_path = f'{problem_dir}/{args.algo}/{args.mode}/{args.exp_set}/{args.seed}/'
    all_contents = os.listdir(directory_path)

    get_intervention_sets = [name for name in all_contents if os.path.isdir(os.path.join(directory_path, name))]
    return get_intervention_sets


def get_problem_names():
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    return sorted(os.listdir(top_dir))


def get_algo_names(args):
    '''
    Get algorithm name / names for comparison, also check if specified algorithm is valid
    '''
    problem_dir = get_problem_dir(args)

    algo_names = set()
    for algo_name in os.listdir(problem_dir):
        if algo_name != 'TrueParetoFront.csv':
            algo_names.add(algo_name)
    if len(algo_names) == 0:
        raise Exception(f'cannot found valid result file under {problem_dir}')

    # if algo argument not specified, return all algorithm names found under the problem directory
    if args.algo is None:
        args.algo = list(algo_names)

    algo_names = args.algo
    return sorted(algo_names)


defaultColors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#fd3216',  # Light24 list
    '#ea1b85',  # more below
    '#7d7803',
    '#ff8fa6',
    '#aeeeee',
    '#7a6ba1',
    '#820028',
    '#d16c6a',
]
