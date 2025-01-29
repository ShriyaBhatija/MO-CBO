import os


def get_intervention_set_name(intervention_variables):
    string = ''
    for i in range(len(intervention_variables)):
        if str(intervention_variables[i]) != 'control':
            string += str(intervention_variables[i]) 
            print(string)
    return string


def get_result_dir(args):
    '''
    Get directory of result location (result/problem/mode/exp_set/seed/)
    '''
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    result_dir = os.path.join(top_dir, args.problem, args.mode, args.exp_set, str(args.seed))
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

    directory_path = f'{problem_dir}/{args.mode}/{args.exp_set}/{args.seed}/'
    all_contents = os.listdir(directory_path)

    get_intervention_sets = [name for name in all_contents if os.path.isdir(os.path.join(directory_path, name))]
    return get_intervention_sets



defaultColours = [
    '#4a90e2',  # lighter muted blue
    '#bc7c75',  # lighter chestnut brown
    '#ff7f0e',  # safety orange
    '#ff3216',  # bright red
    '#d62728',  # brick red
    '#2ca02c',  # cooked asparagus green
    '#17becf',  # blue-teal
    '#e377c2',  # raspberry yogurt pink
    '#bcbd22',  # curry yellow-green
    '#9467bd',  # muted purple
    #'#7f7f7f',  # middle gray
    '#1f77b4',  # muted blue
    '#ea1b85',  # more below
    '#fd3216',  # Light24 list
    '#7d7803',
    '#ff8fa6',
    '#aeeeee',
    '#7a6ba1',
    '#820028',
    '#d16c6a',
]
