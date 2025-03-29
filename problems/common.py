from pymoo.factory import get_from_list
from problems import *
from .graphs.causal_mobo import *


def get_problem_options():

    problems = [
        ('mo-cbo', CausalMOBO)
    ]
    return problems


def get_cbo_options():

    problems = {
        #'mo-cbo1': MO_CBO1(),
        #'mo-cbo2': MO_CBO2(),
        #'mo-cbo3': MO_CBO3(),
        'mo-cbo-health': Health(),
        #'mo-cbo-econ': SCM_Economics(),
    }
    return problems


def get_problem(*args, d={}, **kwargs):
    return get_from_list(get_problem_options(), 'mo-cbo'.lower(), args, {**d, **kwargs})


def build_problem(name, mis):
    '''
    Build optimization problem from name, get initial samples
    ''' 
    try:
        problem = get_problem(graph=get_cbo_options()[name], intervention_set=mis)
    except:
        raise NotImplementedError('problem not supported yet!')
    
    return problem


def calc_causal_pareto_front(name, mis):
    try:
        problem = get_problem(graph=get_cbo_options()[name], intervention_set=mis)
        pareto_front = problem.pareto_front()
    except:
        raise NotImplementedError('problem not supported yet!')
    
    return pareto_front