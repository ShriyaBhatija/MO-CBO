import sys
sys.path.append("..") 

from collections import OrderedDict

from .graph import GraphStructure
from .mocbo6_economic_CostFunctions import define_costs

class SCM_Economics(GraphStructure):
    """
    # Observed variables                    
    # Electricity consumption               Total electricity consumption                                                           0.1 billion kWh
    # Economic growth                       GDP                                                                                     0.1 billion yuan
    # Electricity investment                Fixed capital investments for electricity, thermal power, and natural gas supply        0.1 billion yuan
    # Investments in other industries       The difference between total fixed asset investments and power industry investment      0.1 billion yuan
    # Employment                            Total number of people who are employed                                                 10 thousand
    # Development of the secondary industry Outputs of the secondary industry                                                       0.1 billion yuan
    # Development of the tertiary industry  Outputs of the tertiary industry                                                        0.1 billion yuan
    # Proportion of non-agriculture         Sum of proportions of secondary and tertiary economic sectors                           %
    # Labor productivity                    GDP/employment                                                                          Yuan per capita
    # 
    # Latent variables 
    # Energy source structure               Proportion of renewable energy                                                          %
    # Informatization level                 Number of internet users                                                                10 thousand
    #                                       Number of websites                                                                      10 thousand
    # Ecological awareness                  Investment in environmental protection                                                  0.1 billion yuan
    """

    def define_SEM(self):
        # #TODO check formula ordering respects dependencies
        #
        # Proposed (topologically sorted) ordering:
        #   X1 = U1                                                         (Energy Source Structure)
        #   X4 = U4                                                         (Electricity Consumption; U4 ~ N(0,100000))
        #   X3 = 0.889 * X4 + U4                                            (Ecological Awareness)
        #   X2 = 0.836 * X4 + 0.464 * X3 + U2                               (Informatization Level)
        #   X5 = 0.898 * X4 + U5                                            (Electricity Investment)
        #   X6 = 0.783 * X5 + U6                                            (Investment Other)
        #   X7 = 0.789 * X4 + U7                                            (Employment)
        #   X8 = 0.566 * X4 + 0.561 * X2 + U8                               (Secondary Industry)
        #   X9 = 0.537 * X4 + 0.712 * X2 + U9                               (Tertiary Industry)
        #   X10 = 0.731 * X8 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + U10   (Prop. non-agriculture)
        #   X11 = 0.918 * X4 + U11                                          (Labor Productivity)
        #   Y = 0.538 * X6 + 0.426 * X7 + 0.826 * X11 + 0.293 * X2 +
        #       0.527 * X10 + 0.169 * X4 + 0.411 * X1                       (Output threshold)
        # 
        # Assigning epsilon indices following the topological order:
        def fX1(epsilon, **kwargs):
            # Energy Source Structure
            return epsilon[0]
        
        def fX4(epsilon, **kwargs):
            # Electricity Consumption
            return epsilon[1]
        
        def fX3(epsilon, X4, **kwargs):
            # Ecological Awareness = 0.889 * X4 + U4
            return 0.889 * X4 + epsilon[2]
        
        def fX2(epsilon, X4, X3, **kwargs):
            # Informatization Level = 0.836 * X4 + 0.464 * X3 + U2
            return 0.836 * X4 + 0.464 * X3 + epsilon[3]
        
        def fX5(epsilon, X4, **kwargs):
            # Electricity Investment = 0.898 * X4 + U5
            return 0.898 * X4 + epsilon[4]
        
        def fX6(epsilon, X5, **kwargs):
            # Investment Other = 0.783 * X5 + U6
            return 0.783 * X5 + epsilon[5]
        
        def fX7(epsilon, X4, **kwargs):
            # Employment = 0.789 * X4 + U7
            return 0.789 * X4 + epsilon[6]
        
        def fX8(epsilon, X4, X2, **kwargs):
            # Secondary Industry = 0.566 * X4 + 0.561 * X2 + U8
            return 0.566 * X4 + 0.561 * X2 + epsilon[7]
        
        def fX9(epsilon, X4, X2, **kwargs):
            # Tertiary Industry = 0.537 * X4 + 0.712 * X2 + U9
            return 0.537 * X4 + 0.712 * X2 + epsilon[8]
        
        def fX10(epsilon, X8, X9, X6, X2, **kwargs):
            # Prop. non-agriculture = 0.731 * X8 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + U10
            return 0.731 * X8 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + epsilon[9]
        
        def fX11(epsilon, X4, **kwargs):
            # Labor Productivity = 0.918 * X4 + U11
            return 0.918 * X4 + epsilon[10]
        
        def fY(epsilon, X1, X2, X4, X6, X7, X8, X9, X10, X11, **kwargs):
            # Output threshold = 0.538 * X6 + 0.426 * X7 + 0.826 * X11 + 0.293 * X2 +
            #                    0.527 * X10 + 0.169 * X4 + 0.411 * X1
            return (0.538 * X6 + 0.426 * X7 + 0.826 * X11 +
                    0.293 * X2 + 0.527 * X10 + 0.169 * X4 + 0.411 * X1)
            # (No epsilon term added; if needed, add epsilon[11])
        
        graph = OrderedDict([
            ('X1', fX1),
            ('X4', fX4),
            ('X3', fX3),
            ('X2', fX2),
            ('X5', fX5),
            ('X6', fX6),
            ('X7', fX7),
            ('X8', fX8),
            ('X9', fX9),
            ('X10', fX10),
            ('X11', fX11),
            ('Y',  fY)
        ])
        return graph

    def get_targets(self):
        # The target variable is the output threshold.
        # Interesting Output pairs:
        #   (Energy Sturcutre X1, Economic Growth Y)
        #   (Employment X7, Labour Productivity X11)
        #   (Urbanisation Rate, Employment)
        #   (Secondary Industry, Tertiary Industry)
        #   (Informatization Level X2, Urbanization Rate) 
        #   X1 = U1                                                         (Energy Source Structure)
        #   X4 = U4                                                         (Electricity Consumption; U4 ~ N(0,100000))
        #   X3 = 0.889 * X4 + U4                                            (Ecological Awareness)
        #   X2 = 0.836 * X4 + 0.464 * X3 + U2                               (Informatization Level)
        #   X5 = 0.898 * X4 + U5                                            (Electricity Investment)
        #   X6 = 0.783 * X5 + U6                                            (Investment Other)
        #   X7 = 0.789 * X4 + U7                                            (Employment)
        #   X8 = 0.566 * X4 + 0.561 * X2 + U8                               (Secondary Industry)
        #   X9 = 0.537 * X4 + 0.712 * X2 + U9                               (Tertiary Industry)
        #   X10 = 0.731 * X8 + 0.612 * X9 + 0.662 * X6 + 0.605 * X2 + U10   (Prop. non-agriculture)
        #   X11 = 0.918 * X4 + U11                                          (Labor Productivity)
        #   Y = 0.538 * X6 + 0.426 * X7 + 0.826 * X11 + 0.293 * X2 +
        #       0.527 * X10 + 0.169 * X4 + 0.411 * X1                       (Output threshold)
        # 
        return ['X8', 'X9', 'X10']

    def get_exploration_sets(self):
        # TODO: Define exploration sets for the economic SCM.
        exploration_sets = {
            'mo-cbo': [['X2', 'X4'], ['X6','X2', 'X4']],  # placeholder
            'mobo': ['X2',  'X4', 'X5', 'X6']  # placeholder
        }
        return exploration_sets

    def get_set_MOBO(self):
        # Proposed (topologically sorted) ordering:

        
        # Interesting Intervention set:
        #   (Power Investment X5, Informatization Level X2)
        #   (Other Investment X6, Informatization Level X2)
        #   (Electricity Consumption X4, Informatization Level X2)
        #   X1 = U1                                                         (Energy Source Structure)
        #   X3 = 0.889 * X4 + U4                                            (Ecological Awareness)
        #   X2 = 0.836 * X4 + 0.464 * X3 + U2                               (Informatization Level)
        #   X5 = 0.898 * X4 + U5                                            (Electricity Investment)
        #   X6 = 0.783 * X5 + U6                                            (Investment Other)
        #   (Energy Source Structure X1, Electricity Consumption X4, Power Investment X5, Other Investment X6, Informatization Level X2, Employment X7)
        return ['X2','X4', 'X5', 'X6']

    def get_interventional_ranges(self):
        # TODO: Specify interventional ranges for each manipulable variable.
        dict_ranges = OrderedDict([
            #('X1', [0, 1]),
            ('X2', [0, 1]), # TODO, check whether this is a sound range
            #('X3', [None, None]),
            ('X4', [0, 100000*3]), # TODO: cross check whether this makes sense for china's energy grid
            ('X5', [0, 18000]),
            ('X6', [0, 18000]),
            # ('X7', [0, 1400000000/10000]),
            # ('X8', [None, None]),
            # ('X9', [None, None]),
            # ('X10', [None, None]),
            # ('X11', [None, None])
        ])
        return dict_ranges

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs
