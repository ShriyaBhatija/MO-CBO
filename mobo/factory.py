"""
This code was originally published by the following individual:
    Copyright (c) 2020 Yunsheng Tian
    GitHub: https://github.com/yunshengtian/DGEMO?tab=MIT-1-ov-file
"""


'''
Factory for importing different components of the MOBO framework by name
'''

def get_surrogate_model(name):
    from .surrogate_model import GaussianProcess
    
    surrogate_model = {'gp': GaussianProcess}

    surrogate_model['default'] = GaussianProcess

    return surrogate_model[name]


def get_acquisition(name):
    from .acquisition import IdentityFunc

    acquisition = {'identity': IdentityFunc}

    acquisition['default'] = IdentityFunc

    return acquisition[name]


def get_solver(name):
    from .solver import ParetoDiscoverySolver

    solver = {'discovery': ParetoDiscoverySolver}

    solver['default'] = ParetoDiscoverySolver

    return solver[name]


def get_selection(name):
    from .selection import CPS_Select

    selection = {'cps': CPS_Select}

    selection['default'] = CPS_Select

    return selection[name]


def init_from_config(config, framework_args):
    '''
    Initialize each component of the MOBO framework from config
    '''
    init_func = {
        'surrogate': get_surrogate_model,
        'acquisition': get_acquisition,
        'selection': get_selection,
        'solver': get_solver,
    }

    framework = {}
    for key, func in init_func.items():
        kwargs = framework_args[key]
        if config is None:
            # no config specified, initialize from user arguments
            name = kwargs[key]
        else:
            # initialize from config specifications, if certain keys are not provided, use default settings
            name = config[key] if key in config else 'default'
        framework[key] = func(name)(**kwargs)

    return framework