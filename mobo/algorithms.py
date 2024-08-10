from .mobo import MOBO

'''
High-level algorithm specifications by providing config
'''

class DGEMO(MOBO):
    '''
    DGEMO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'identity',
        'solver': 'discovery',
        'selection': 'dgemo',
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
        'dgemo': DGEMO,
        'custom': Custom,
    }
    return algo[name]