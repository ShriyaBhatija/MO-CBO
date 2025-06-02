"""
This code was originally published by the following individual:
    Copyright (c) 2020 Yunsheng Tian
    GitHub: https://github.com/yunshengtian/DGEMO?tab=MIT-1-ov-file
"""


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


class TSEMO(MOBO):
    '''
    TSEMO
    '''
    config = {
        'surrogate': 'ts',
        'acquisition': 'identity',
        'solver': 'nsga2',
        'selection': 'hvi',
    }

class ParEGO(MOBO):
    '''
    ParEGO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'parego',
        'selection': 'random',
    }

class MOEAD_EGO(MOBO):
    '''
    MOEA/D-EGO
    '''
    config = {
        'surrogate': 'gp',
        'acquisition': 'ei',
        'solver': 'moead',
        'selection': 'moead',
    }


def get_algorithm(name):
    '''
    Get class of algorithm by name
    '''
    algo = {
        'dgemo': DGEMO,
        'tsemo': TSEMO,
        'moead-ego': MOEAD_EGO,
        'parego': ParEGO
    }
    return algo[name]