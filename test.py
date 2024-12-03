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
n_pareto_points = 10
x1_range = [5.0]
x2_range = anp.linspace(0, 5.0, 10000)
print(x2_range)


points = [[5.0, val] for val in x2_range]

x1_grid, x2_grid = anp.meshgrid(x1_range, x2_range)
points = anp.vstack([x1_grid.ravel(), x2_grid.ravel()]).T

