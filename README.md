# MP-Plasmo Benders Optimization Code

This repository contains the code used in our manuscript for solving an optimization problem using mp-Plasmo Benders.

## 🔧 Instructions to Run

### Step 1: Generate the mp Subproblem Solution

Before running the main code, you must generate the mp solution, which is done in Python. 

Go to the `mp_sub_problem/` folder and run the following file in Python:

```python
generate_mp_solution.ipynb
```

This will generate the mp subproblem solution required for the main code.

This code was run with the following versions: 
 - 

### Step 2: Run the Main Code

After generating the mp solution, the code in the `main_code/` folder can be run in Julia. This folder includes an environment (`main_code/env`). If you are not familiar with environments, you can read about them [here](https://pkgdocs.julialang.org/v1/environments/). 

The environment can be activated by starting the Julia REPL, typing `]` (to enter the Package manager) and typing

```
pkg> activate ./main_code/env
```

The environment can then be activated by typing 

```
pkg> instantiate
```

This will install the required packages using the same versions used in this work. 

The file `main_code/main_code.jl` is the primary script to run to replicate our results. this directory also includes the following files or subdirectories: 
 - `PlasmoAlgorithms.jl` - this includes source code for running `PlasmoBenders.jl`.`PlasmoBenders.jl` is a registered package, and the framework outlined here can be run with the registered version. However, this source code includes minor additions which allow for timing the master problem and subproblem solves within Benders (used for comparing the time to solve with the custom MP solver in the subproblem compared to the subproblem solves using traditional solvers like Gurobi or HiGHS). This source code is checked out for development in the included environment, so the user does not need to interact with this folder if they instantiate the included environment folder. 
 - `MPBenders_solver_primal_minimal.jl` and `mp_struct_primal.jl` - these files include the code for running the custom solver used in the `main_code.jl`. Much of this code extends MathOptInterface.jl (MOI) functionality and was adapted from [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl)'s source code. This code allows for defining a custom solver on a JuMP model or Plasmo OptiGraph and extending the methods for querying primal and dual information (e.g., extending the MOI internals called by `JuMP.value` or `JuMP.dual`) to use the MP data. 



## 📊 Visualization

Visualizations can also be generated from the jupyter notebooks in the `main_code/` directory, after first running the `main_code/File_run_for_CR_plots.jl`.

- To **plot the critical regions**, run:

```python
visual_CR.ipynb
```

- To **plot the runtime**, run:

```python
time_plot.ipynb
```
