import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict


## Define a cost variable for each intervention
def cost_bmi_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_statin_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_aspirin_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_control_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_equal),
        ('statin', cost_statin_fix_equal),
        ('aspirin', cost_aspirin_fix_equal),
        ('control', cost_control_fix_equal),
            ])
    
    return costs