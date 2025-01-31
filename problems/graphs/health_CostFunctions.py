import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict


## Define a cost variable for each intervention
def cost_bmi_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_weight_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_ci_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_aspirin_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost


def cost_bmi_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost

def cost_weight_fix_different(intervention_value, **kwargs):
    fix_cost = 2.
    return fix_cost

def cost_ci_fix_different(intervention_value, **kwargs):
    fix_cost = 8.
    return fix_cost

def cost_aspirin_fix_different(intervention_value, **kwargs):
    fix_cost = 5.
    return fix_cost


def cost_bmi_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_weight_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 2.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_ci_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 8.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_aspirin_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 5.
    return np.sum(np.abs(intervention_value)) + fix_cost



def cost_bmi_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_weight_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_ci_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_aspirin_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost



def define_costs(type_cost):
    
    if type_cost == 1:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_equal),
        ('weight', cost_weight_fix_equal),
        ('ci', cost_ci_fix_equal),
        ('aspirin', cost_aspirin_fix_equal)
            ])
        
    if type_cost == 2:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_different),
        ('weight', cost_weight_fix_different),
        ('ci', cost_ci_fix_different),
        ('aspirin', cost_aspirin_fix_different)
            ])
        
    if type_cost == 3:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_different_variable),
        ('weight', cost_weight_fix_different_variable),
        ('ci', cost_ci_fix_different_variable),
        ('aspirin', cost_aspirin_fix_different_variable)
            ])
    
    if type_cost == 4:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_equal_variable),
        ('weight', cost_weight_fix_equal_variable),
        ('ci', cost_ci_fix_equal_variable),
        ('aspirin', cost_aspirin_fix_equal_variable)  
            ])
        
    return costs