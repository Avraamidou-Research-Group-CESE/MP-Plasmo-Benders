# MP-Plasmo Benders Optimization Code

This repository contains the code used in our manuscript for solving an optimization problem using mp-Plasmo Benders.

## 🔧 Instructions to Run

### Step 1: Generate the mp Subproblem Solution

Before running the main code, you must generate the mp solution.

Go to the `mp_sub_problem/` folder and run the following file in Python:

```python
generate_mp_solution.ipynb
```

This will generate the mp subproblem solution required for the main code.

### Step 2: Run the Main Code

After generating the mp solution, go to the `main_code/` folder.

Run the main file:

```julia
main_code.jl
```

**Note:**  
Make sure you run this in a Julia environment where the updated `PlasmoAlgorithm.jl` is available.  
We have provided this in the file `PlasmoAlgorithm.jl.zip`.

## 📊 Visualization

- To **plot the critical regions**, run:

```python
visual_CR.ipynb
```

- To **plot the runtime**, run:

```python
time_plot.ipynb
```
