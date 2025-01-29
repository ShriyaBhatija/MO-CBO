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

def cost_statin_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_aspirin_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost


## Define a cost variable for each intervention
def cost_bmi_fix_different(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_statin_fix_different(intervention_value, **kwargs):
    fix_cost = 10.
    return fix_cost

def cost_aspirin_fix_different(intervention_value, **kwargs):
    fix_cost = 5.
    return fix_cost


## Define a cost variable for each intervention
def cost_bmi_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_statin_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 10.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_aspirin_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 5.
    return np.sum(np.abs(intervention_value)) + fix_cost



## Define a cost variable for each intervention
def cost_bmi_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_statin_fix_equal_variable(intervention_value, **kwargs):
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
        ('statin', cost_statin_fix_equal),
        ('aspirin', cost_aspirin_fix_equal)
            ])
        
    if type_cost == 2:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_different),
        ('statin', cost_statin_fix_different),
        ('aspirin', cost_aspirin_fix_different)
            ])
        
    if type_cost == 3:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_different_variable),
        ('statin', cost_statin_fix_different_variable),
        ('aspirin', cost_aspirin_fix_different_variable)
            ])
    
    if type_cost == 4:
        costs = OrderedDict ([
        ('bmi', cost_bmi_fix_equal_variable),
        ('statin', cost_statin_fix_equal_variable),
        ('aspirin', cost_aspirin_fix_equal_variable)  
            ])
        
    return costs