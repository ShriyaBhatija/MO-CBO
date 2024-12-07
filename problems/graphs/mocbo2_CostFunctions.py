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

def cost_X4_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X5_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X6_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X7_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_X8_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost



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

def cost_X4_fix_different(intervention_value, **kwargs):
    fix_cost = 20.
    return fix_cost

def cost_X5_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost

def cost_X6_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost

def cost_X7_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost

def cost_X8_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost



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

def cost_X4_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 20.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X5_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X6_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X7_fix_different_variable(intervention_value, **kwargs):    
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X8_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost



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

def cost_X4_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X5_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X6_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X7_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_X8_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost


def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_equal),
        ('X2', cost_X2_fix_equal),
        ('X3', cost_X3_fix_equal),
        ('X4', cost_X4_fix_equal),
        ('X5', cost_X5_fix_equal),
        ('X6', cost_X6_fix_equal),
        ('X7', cost_X7_fix_equal),
        ('X8', cost_X8_fix_equal)
            ])
        
    if type_cost == 2:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_different),
        ('X2', cost_X2_fix_different),
        ('X3', cost_X3_fix_different),
        ('X4', cost_X4_fix_different),
        ('X5', cost_X5_fix_different),
        ('X6', cost_X6_fix_different),
        ('X7', cost_X7_fix_different),
        ('X8', cost_X8_fix_different)
            ])

    if type_cost == 3:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_different_variable),
        ('X2', cost_X2_fix_different_variable),
        ('X3', cost_X3_fix_different_variable),
        ('X4', cost_X4_fix_different_variable),
        ('X5', cost_X5_fix_different_variable),
        ('X6', cost_X6_fix_different_variable),
        ('X7', cost_X7_fix_different_variable),
        ('X8', cost_X8_fix_different_variable)
            ])

    if type_cost == 4:
        costs = OrderedDict ([
        ('X1', cost_X1_fix_equal_variable),
        ('X2', cost_X2_fix_equal_variable),
        ('X3', cost_X3_fix_equal_variable),
        ('X4', cost_X4_fix_equal_variable),
        ('X5', cost_X5_fix_equal_variable),
        ('X6', cost_X6_fix_equal_variable),
        ('X7', cost_X7_fix_equal_variable),
        ('X8', cost_X8_fix_equal_variable)
            ])

    return costs