import numpy as np
from collections import OrderedDict

def make_cost_function(fixed_cost, include_variable=False):
    """
    Returns a cost function that adds a fixed cost,
    and optionally a variable cost (sum of absolute values).
    """
    def cost(intervention_value, **kwargs):
        result = fixed_cost
        if include_variable:
            result += np.sum(np.abs(intervention_value))
        return result
    return cost

def define_costs(type_cost):
    """
    Define cost functions based on the cost type.
    
    Type mapping:
      1: All interventions have fixed cost 1.
      2: Different fixed costs for each intervention.
      3: Same as type 2, but adds a variable cost component.
      4: All interventions have fixed cost 1, plus a variable cost.
    """
    # Define a configuration for each cost type:
    cost_config = {
        1: {"costs": {"X11": 0.5, "X12": 0.5, "X2": 1, "X5": 1, "X6":1, "X3":1}, "variable": False},
        2: {"costs": {"X11": 1, "X12": 1, "X2": 5, "X5": 7, "X6":9, "X3":11}, "variable": False},
        3: {"costs": {"X11": 1, "X12": 1, "X2": 5, "X5": 7, "X6":9, "X3":11}, "variable": True},
        4: {"costs": {"X11": 0.5, "X12": 0.5, "X2": 1, "X5": 1, "X6":1, "X3":1}, "variable": True},
    }
            #    ('X11', [0, 100]),
            # ('X12', [0, 100]),
            # ('X2', [0, 100]),
            # ('X3', [0, 100]),
            # ('X5', [0, 50]),
            # ('X6', [0, 5000]),
            # ('X3', [0, 100]),
    
    config = cost_config.get(type_cost)
    if config is None:
        raise ValueError(f"Unknown cost type: {type_cost}")

    costs = OrderedDict()
    for key, fixed_cost in config["costs"].items():
        costs[key] = make_cost_function(fixed_cost, include_variable=config["variable"])
    
    return costs
