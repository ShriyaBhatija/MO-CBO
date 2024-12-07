'''
def solution(S):
  
  codes = []
  
  # Iterate through all codes
  for i_1 in range(0,10):
    for i_2 in range(0,10):
      for i_3 in range(0,10):
        for i_4 in range(0,10):
          code = ""
          if i_1 + i_2 +i_3 +i_4 == S:
            code += str(i_1) + str(i_2) + str(i_3) + str(i_4)
            codes.append(code)
    
  
  return len(codes)

'''
import autograd.numpy as anp
import numpy as np
from numpy.random import randn
#from pymoo.problems.multi import *
#from pymoo.visualization.scatter import Scatter

# The pareto front of a scaled zdt1 problem
#pf = ZDT1().pareto_front()
#print(pf)

  #x1_range = anp.linspace(xl[0], xu[0], n_pareto_points)

'''
n_pareto_points = 10
x1_range = [5.0]
x2_range = anp.linspace(0, 5.0, 10000)
print(x2_range)


points = [[5.0, val] for val in x2_range]

x1_grid, x2_grid = anp.meshgrid(x1_range, x2_range)
points = anp.vstack([x1_grid.ravel(), x2_grid.ravel()]).T
'''

'''
import pandas as pd
import os

  
#observational data
for seed in range(0,10):
  data = pd.read_pickle(f'./Data/mo-cbo1/mis/{seed}/observations_complete.pkl')
  mean_Y1 = np.mean(np.asarray(data['Y1']))
  mean_Y2 = np.mean(np.asarray(data['Y2']))

  # create folder
  path = f'./result/mo-cbo1//cps/int_data/pomis/{seed}/empty'
  os.makedirs(path, exist_ok=True)

  # Define the CSV file path
  csv_file_path = os.path.join(path, 'sample.csv')

  # Create a DataFrame with the specified columns
  df = pd.DataFrame({'Pareto_f1': [mean_Y1], 'Pareto_f2': [mean_Y2]}) 

  # Save the DataFrame to a CSV file
  df.to_csv(csv_file_path, index=False) 
'''

import pandas as pd

#interventional_data = np.load(f'./Data/mo-cbo-coral/mobo/1/interventional_data.npy', allow_pickle=True)
#interventional_data = np.load(f'./Data/mo-cbo1/mobo/0/interventional_data.npy', allow_pickle=True)
observational_data = pd.read_pickle(f'./Data/mo-cbo2/pomis/0/observations.pkl')
#print(interventional_data)
#print(interventional_data_new[1][3])
#print(interventional_data[1][3])

print(observational_data)