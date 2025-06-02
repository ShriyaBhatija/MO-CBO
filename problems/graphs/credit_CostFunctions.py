
import numpy as np
from collections import OrderedDict

def cost_X3_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X4_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X5_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X6_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost


def define_costs(type_cost):
    
    if type_cost == 1:
        costs = OrderedDict ([
        ('X3', cost_X3_fix_equal),
        ('X4', cost_X4_fix_equal),
        ('X6', cost_X5_fix_equal),
        ('X7', cost_X6_fix_equal)
        ])
        
    return costs