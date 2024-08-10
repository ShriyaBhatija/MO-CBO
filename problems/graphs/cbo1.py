import sys
sys.path.append("..") 

from collections import OrderedDict
import autograd.numpy as anp
import numpy as np
import pandas as pd

from .graph import GraphStructure
from utils_functions.utils import fit_single_GP_model

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .cbo1_DoFunctions import *


class CBO1(GraphStructure):
    
    def __init__(self, observational_samples):
                
        self.X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        self.X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        self.Z1 = np.asarray(observational_samples['Z1'])[:,np.newaxis]
        self.Z2 = np.asarray(observational_samples['Z2'])[:,np.newaxis]
        self.Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        self.Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]


    def define_SEM(self):

        def fx1(epsilon, **kwargs):
          return epsilon[0]
        
        def fx2(epsilon, **kwargs):
          return epsilon[1]
        
        def fz1(epsilon, X1, X2, **kwargs):
          return ((X1-X2)/2)**3
        
        def fz2(epsilon, X1, X2, **kwargs):
          return anp.exp((X1-X2)/2) 

        def fy1(epsilon, Z1, Z2, **kwargs):
          return (Z1+Z2)**2

        def fy2(epsilon, Z1, Z2, **kwargs):
          return (Z1+Z2-10)**2

        graph = OrderedDict ([
          ('X1', fx1),
          ('X2', fx2),
          ('Z1', fz1),
          ('Z2', fz2),
          ('Y1', fy1),
          ('Y2', fy2),
        ])

        return graph
    

    def get_targets(self):
        return ['Y1', 'Y2'] 
    

    def get_sets(self):
      MIS = [['X1', 'X2'], ['Z1','Z2']]
      manipulative_variables = ['X1', 'X2', 'Z1', 'Z2']
      return MIS, manipulative_variables
    

    def get_set_BO(self):
      manipulative_variables = ['X1', 'X2', 'Z1', 'Z2']
      return manipulative_variables
    

    def get_interventional_ranges(self):
      min_intervention_x = -1
      max_intervention_x = 2

      min_intervention_z = -1
      max_intervention_z = 2

      dict_ranges = OrderedDict ([
          ('X1', [min_intervention_x, max_intervention_x]),
          ('X2', [min_intervention_x, max_intervention_x]),
          ('Z1', [min_intervention_z, max_intervention_z]),
          ('Z2', [min_intervention_z, max_intervention_z])
        ])
      
      return dict_ranges
    

    def fit_all_models(self):

        functions = {}
        inputs_list = [ 
                        [self.X1], 
                        [self.X2],
                        [self.Z1, self.Z2, np.hstack((self.X1,self.Z1)), np.hstack((self.X2,self.Z2)), np.hstack((self.Z1,self.Z2)), np.hstack((self.X1,self.X2,self.Z1,self.Z2))],
                        [self.Z1, self.Z2, np.hstack((self.X1,self.Z1)), np.hstack((self.X2,self.Z2)), np.hstack((self.Z1,self.Z2)), np.hstack((self.X1,self.X2,self.Z1,self.Z2))]
                        ]
        output_list = [
                        self.Z1,
                        self.Z2, 
                        self.Y1, 
                        self.Y2
                      ]
        name_list = [ 
                      ['gp_X1_toZ1'],
                      ['gp_X2_toZ2'], 
                      ['gp_Z1_toY1', 'gp_Z2_toY1', 'gp_X1_Z1_toY1', 'gp_X2_Z2_toY1', 'gp_Z1_Z2_toY1', 'gp_X1_X2_Z1_Z2_toY1'],
                      ['gp_Z1_toY2', 'gp_Z2_toY2', 'gp_X1_Z1_toY2', 'gp_X2_Z2_toY2', 'gp_Z1_Z2_toY2', 'gp_X1_X2_Z1_Z2_toY2']
                      ]      
        parameter_list = [[1.,1.,1., False], 
                          [1.,1.,1., False], 
                          [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False],
                          [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False]
                        ]
        
        ## Fit all conditional models
        for j in range(len(output_list)):
            for k in range(len(inputs_list[j])): 
                  X = inputs_list[j][k]
                  Y = output_list[j]
                  parameters = parameter_list[j+k]
                  print(name_list[j][k])
                  functions[name_list[j][k]] = fit_single_GP_model(X, Y, parameters)
    
        return functions

  

    def refit_models(self, observational_samples):
        X1 = np.asarray(observational_samples['X1'])[:,np.newaxis]
        X2 = np.asarray(observational_samples['X2'])[:,np.newaxis]
        Z1 = np.asarray(observational_samples['Z1'])[:,np.newaxis]
        Z2 = np.asarray(observational_samples['Z2'])[:,np.newaxis]
        Y1 = np.asarray(observational_samples['Y1'])[:,np.newaxis]
        Y2 = np.asarray(observational_samples['Y2'])[:,np.newaxis]

        functions = {}
        inputs_list = [ 
                        [X1], 
                        [X2],
                        [Z1, Z2, np.hstack((X1,Z1)), np.hstack((X2,Z2)), np.hstack((Z1,Z2)), np.hstack((X1,X2,Z1,Z2))],
                        [Z1, Z2, np.hstack((X1,Z1)), np.hstack((X2,Z2)), np.hstack((Z1,Z2)), np.hstack((X1,X2,Z1,Z2))]
                        ]
        output_list = [
                        Z1,
                        Z2, 
                        Y1, 
                        Y2
                      ]
        name_list = [ 
                      ['gp_X1_toZ1'],
                      ['gp_X2_toZ2'], 
                      ['gp_Z1_toY1', 'gp_Z2_toY1', 'gp_X1_Z1_toY1', 'gp_X2_Z2_toY1', 'gp_Z1_Z2_toY1', 'gp_X1_X2_Z1_Z2_toY1'],
                      ['gp_Z1_toY2', 'gp_Z2_toY2', 'gp_X1_Z1_toY2', 'gp_X2_Z2_toY2', 'gp_Z1_Z2_toY2', 'gp_X1_X2_Z1_Z2_toY2']
                      ]      
        parameter_list = [1.,1.,1., False]
        
        ## Fit all conditional models
        for j in range(len(output_list)):
            for k in range(len(inputs_list[j])): 
                  X = inputs_list[j][k]
                  Y = output_list[j]
                  functions[name_list[j][k]] = fit_single_GP_model(X, Y, parameter_list)
    
        return functions



    def get_all_do(self):
        do_dict = {}
        do_dict['compute_do_X1'] = compute_do_X1
        do_dict['compute_do_X2'] = compute_do_X2
        do_dict['compute_do_X1X2'] = compute_do_X1X2
        do_dict['compute_do_Z1'] = compute_do_Z1
        do_dict['compute_do_Z2'] = compute_do_Z2
        do_dict['compute_do_Z1Z2'] = compute_do_Z1Z2
        return do_dict
  