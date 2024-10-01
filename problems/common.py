import numpy as np
from pymoo.factory import get_from_list, get_reference_directions
from problems import *
from external import lhs
from .graphs.cbo1 import *
from .graphs.causal_mobo import *


def get_problem_options():

    problems = [
        ('cbo', CausalMOBO)
    ]
    return problems


def get_cbo_options(observational_samples):

    problems = {
        'cbo1': CBO1(observational_samples)
    }
    return problems


def get_problem(name, *args, d={}, **kwargs):
    if name.startswith('cbo'):
        return get_from_list(get_problem_options(), 'cbo'.lower(), args, {**d, **kwargs})
    else:
        return get_from_list(get_problem_options(), name.lower(), args, {**d, **kwargs})


def generate_initial_samples(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    X_feasible = np.zeros((0, problem.n_var))
    Y_feasible = np.zeros((0, problem.n_obj))

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    while len(X_feasible) < n_sample:
        X = lhs(problem.n_var, n_sample)
        X = problem.xl + X * (problem.xu - problem.xl)
        #X = list(np.linspace(problem.xl, problem.xu, n_sample))

        Y, feasible = problem.evaluate(X, return_values_of=['F', 'feasible'])
        feasible = feasible.flatten()
        X_feasible = np.vstack([X_feasible, X[feasible]])
        Y_feasible = np.vstack([Y_feasible, Y[feasible]])
    
    indices = np.random.permutation(np.arange(len(X_feasible)))[:n_sample]
    X, Y = X_feasible[indices], Y_feasible[indices]
    return X, Y


def build_problem(name, observational_samples, n_var, n_obj, n_init_sample, mis=None, n_process=1):
    '''
    Build optimization problem from name, get initial samples
    Input:
        name: name of the problem (supports ZDT1-6, DTLZ1-7)
        n_var: number of design variables
        n_obj: number of objectives
        n_init_sample: number of initial samples
        n_process: number of parallel processes
    Output:
        problem: the optimization problem
        X_init, Y_init: initial samples
        pareto_front: the true pareto front of the problem (if defined, otherwise None)
    '''
    # build problem
    if name.startswith('cbo'):
        assert mis is not None, 'intervention set must be provided for CBO problems'
        problem = get_problem(name, graph=get_cbo_options(observational_samples)[name], intervention_set=mis)
        print(problem)
        pareto_front = None

    else:
        try:
            problem = get_problem(name)
        except:
            raise NotImplementedError('problem not supported yet!')
        try:
            pareto_front = problem.pareto_front()
        except:
            pareto_front = None

    X_init, Y_init = generate_initial_samples(problem, n_init_sample)
    
    return problem, pareto_front, X_init, Y_init