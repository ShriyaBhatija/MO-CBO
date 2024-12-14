import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict


## Define a cost variable for each intervention
def cost_X1_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X2_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X3_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_control_fix_equal(intervention_value, **kwargs):
    return 0.



## Define a cost variable for each intervention
def cost_X1_fix_different(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X2_fix_different(intervention_value, **kwargs):
    fix_cost = 10.
    return fix_cost

def cost_X3_fix_different(intervention_value, **kwargs):
    fix_cost = 5.
    return fix_cost

def cost_control_fix_different(intervention_value, **kwargs):
    return 0.



## Define a cost variable for each intervention
def cost_X1_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X2_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 10.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X3_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 5.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_control_fix_different_variable(intervention_value, **kwargs):
    return 0.



## Define a cost variable for each intervention
def cost_X1_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X2_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X3_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_control_fix_equal_variable(intervention_value, **kwargs):
    return 0.



def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_equal),
        ('X2', cost_X2_fix_equal),
        ('X3', cost_X3_fix_equal),
        ('control', cost_control_fix_equal)
            ])
        
    if type_cost == 2:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_different),
        ('X2', cost_X2_fix_different),
        ('X3', cost_X3_fix_different),
        ('control', cost_control_fix_different)
            ])

    if type_cost == 3:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_different_variable),
        ('X2', cost_X2_fix_different_variable),
        ('X3', cost_X3_fix_different_variable),
        ('control', cost_control_fix_different_variable)
            ])

    if type_cost == 4:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_equal_variable),
        ('X2', cost_X2_fix_equal_variable),
        ('X3', cost_X3_fix_equal_variable),
        ('control', cost_control_fix_equal_variable)
            ])

    return costs