# Multi-Objective Causal Bayesian Optimization (MO-CBO)
Multi-Objective Causal Bayesian Optimization proposes a new class of optimization problems, that aims to discover Pareto-optimal interventions in multi-outcome causal graphs. 

![assets/mo_cbo_visual.pdf](assets/mo_cbo_visual.pdf)

<object data="assets/mo_cbo_visual.pdf" width="1000" height="1000" type='application/pdf'/>



## Code structure
````
external/ --- lhs funtion for sampling
helpers/ --- helper functions
mobo/
 ├── solver/ --- multi-objective solvers
 ├── surrogate_model/ --- surrogate models
 ├── acquisition.py --- acquisition functions
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── mobo.py --- main pipeline of multi-objective bayesian optimziation
 ├── selection.py --- selection methods for new samples
 ├── surrogate_problem.py --- multi-objective surrogate problem
 ├── transformation.py --- normalizations on data
 └── utils.py --- utility functions
problems/
 ├── graphs/ --- mo-cbo problems
 ├── common.py --- common functions
 └── problems.py --- Problem class
visualization/ --- performance visualization
create_datasets.py --- dataset creation for the mo-cbo problems
C_ParetoSelect.py --- mo-cbo algorithm 
main.py --- main execution file for the mo-cbo algorithm
````

## Requirements
• Python version: tested in Python 3.7.7
• Operating system: tested in Ubuntu 18.04

Install the environment by conda and activate:




## Custom Problem Setup 


## Citation
Master's thesis jointly conducted at the Technical University of Munich and University of Cambridge. If you find our repository helpful for your work, please cite our paper:
TBA 
