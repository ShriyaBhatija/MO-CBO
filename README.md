# Multi-Objective Causal Bayesian Optimisation (MO-CBO)
This repository is associated with the Master's thesis of [Shriya Bhatija](https://www.linkedin.com/in/shriya-bhatija-565699155), conducted under joint supervision at the University of Cambridge and the Technical University of Munich (TUM).

[[Thesis]](assets/THESIS.pdf)

Supervisor: [Matthias Althoff](https://www.ce.cit.tum.de/cps/members/prof-dr-ing-matthias-althoff/)

Advisors: [Jakob Thumm](https://jakob-thumm.com), [Paul-David Zuercher](https://pauldavidzuercher.com), [Thomas Bohné](https://www.ifm.eng.cam.ac.uk/people/tmb35/)

#### Abstract
We propose multi-objective causal Bayesian optimisation (MO-CBO), a new problem class for identifying Pareto-optimal interventions that simultaneously optimise multiple target variables within a known causal graph. MO-CBO extends the [causal Bayesian optimisation (CBO)](https://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf) family of methods to support optimisation on causal models with multiple outcomes. We prove that any mo-cbo problem can be decomposed into a series of traditional multi-objective optimisation tasks, and introduce Causal ParetoSelect, an algorithm that sequentially balances exploration across these tasks using relative hypervolume improvement. Our methodology generalises multi-objective Bayesian optimisation to perform causally-informed function evaluations, instead of neglecting known causal relationships. By establishing graphical criteria, we enforce Causal ParetoSelect to explore only potentially optimal sets of variables to intervene upon. We validate our approach on both synthetic and real-world causal graphs, demonstrating its superiority over traditional multi-objective Bayesian optimisation.

#### Methodology Overview
<img src="assets/mo_cbo_visual.png" width="1000">

## Code structure
````
external/ --- lhs funtion for sampling
helpers/ --- helper functions for graph operations
mobo/ --- multi-objective Bayesian optimisation algorithm
 ├── solver/ --- multi-objective solvers
 ├── surrogate_model/ --- surrogate models
 ├── acquisition.py --- acquisition functions
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── mobo.py --- main pipeline of multi-objective bayesian optimsiation
 ├── selection.py --- selection methods for new samples
 ├── surrogate_problem.py --- multi-objective surrogate problem
 ├── transformation.py --- normalisations on data
 └── utils.py --- utility functions
problems/
 ├── graphs/ --- mo-cbo problems
 ├── common.py --- common functions
 └── problems.py --- problem class
visualization/ --- performance visualization
create_datasets.py --- dataset creation for the mo-cbo problems
C_ParetoSelect.py --- mo-cbo algorithm 
main.py --- main execution file for the mo-cbo algorithm
````

## Requirements
Python version: tested in Python 3.7.7

Operating system: tested in macOS Sonoma 14.4.1

Install docker [docker](https://docs.anaconda.com/miniconda/) to use the environment.

We provide a docker image in a public repository:
```bash
docker pull pauldavidzuercher/mocbo
docker tag pauldavidzuercher/mocbo mocbo
```

Alternatively, you can build it from source:
```bash
docker build . -t mocbo
```

To run the example optimisation use:
````bash
docker run -t mocbo && conda activate mo_cbo
````

## Getting started

You can entry the docker environment using:
```bash
docker run -i --entrypoint /bin/bash -t mocbo
```

Subsequently, run the main file with some specified arguments, e.g. problem name, exploration set, batch size and seed:
````bash
python main.py --problem mo-cbo1 --exp-set mobo --batch-size 5 --seed 0
````
For more arguments, we refer to *arguments.py*. The results of this experiment will be stored in ` result/mo-cbo1/int_data/mobo/0/`.
For demonstration purposes, we run this experiment on the problem mo-cbo2, repeating it for both exploration sets, which are *mobo* (baseline) and *mo-cbo* (ours). Note that mo-cbo1 and mo-cbo2 are synthetic structural causal models that were specifically designed for this new type of problem class. We visualise the results by running:
````
python visualize/visualize_pf_all.py --problem mo-cbo1 --seed 0
````
The resulting Pareto front visualisations are:

<img src="assets/pf_visual.png" width="750">

We repeat these experiments across 10 random seeds to report averaged performance metrics that can be visualised by running:
````bash
python visualize/plots_gd_cost.py --problem mo-cbo1 --metric gd
python visualize/plots_gd_cost.py --problem mo-cbo1 --metric igd
````
The resulting visualisations are:

<img src="assets/performances.png" width="500">

## Custom Problem Setup 

If you are interested in implementing your own custom problem, please do the following steps:

1. Create the file `problems/graphs/myproblem.py` where the structural causal model will be defined. The function `define_SEM()` defines the structural equations between the nodes, `get_targets()` returns the target variables and `get_exploration_sets()` returns the exploration sets for the baseline as well as MO-CBO. Moreover, `get_set_MOBO()` gives all manipulative variables, `get_interventional_ranges()` specifies the domains of the interventions and `get_cost_structure()` defines the penality or cost for each intervention performed.

````python
from collections import OrderedDict
import autograd.numpy as anp
from .graph import GraphStructure

class MyProblem(GraphStructure):
    def define_SEM(self):
        def fx1(epsilon, **kwargs):
          return epsilon[0]
        def fx2(epsilon, X1, **kwargs):
          return X1 + epsilon[1]
        def fy1(epsilon, X2, **kwargs):
          return X2 + epsilon[2]
        def fy2(epsilon, X2, **kwargs):
          return X2 - epsilon[3]

        graph = OrderedDict ([
          ('X1', fx1),
          ('X2', fx2),
          ('Y1', fy1),
          ('Y2', fy2),
        ])
        return graph
    
    def get_targets(self):
        return ['Y1', 'Y2'] 
    
    def get_exploration_sets(self):
      exploration_sets = {'mo-cbo': [['X2']],'mobo': [['X1', 'X2']]}
      return exploration_sets

    def get_set_MOBO(self):
      return ['X1', 'X2']
    
    def get_interventional_ranges(self):
      dict_ranges = OrderedDict ([
          ('X1', [-1, 1]),
          ('X2', [2, 4]) ])
      return dict_ranges  
    
    def get_cost_structure(self, type_cost):
        costs = OrderedDict ([('X1', 1),('X2', 1),])
        return costs
````

2. In `problems/__init__.py`, add the line `.graphs.myproblem import MyProblem`
3. In `problems/common.py`, append a tuple `('myproblem', MyProblem)` to the problems variable in `get_problem_options()` such that this problem is callable from command line arguments
4. Create the observational and interventional datasets that will be used in the algorithm. Here, run for example
  ````bash
  python3 create_datasets.py --problem myproblem --exp-set mobo --seed 0
  ````
to save the data in the folder `Data/myproblem/mobo/0`.
 
