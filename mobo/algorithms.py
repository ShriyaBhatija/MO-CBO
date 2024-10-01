from .mobo import MOBO

'''
High-level algorithm specifications by providing config
'''

class Causal_ParetoSelect(MOBO):
    '''
    Causal ParetoSelect (CPS)
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'identity',
        'solver': 'discovery',
        'selection': 'cps',
    }


'''
Define new algorithms here
'''


class Custom(MOBO):
    '''
    Totally rely on user arguments to specify each component
    '''
    config = None


def get_algorithm(name):
    '''
    Get class of algorithm by name
    '''
    algo = {
        'cps': Causal_ParetoSelect,
        'custom': Custom,
    }
    return algo[name]