using Revise
using JuMP
using Gurobi
using Plasmo
using PlasmoBenders
using HiGHS
using Interpolations
using Suppressor
using DelimitedFiles
using CSV
using DataFrames

include((@__DIR__)*"/MPBenders_solver_primal_minimal.jl")

sub_solver_Gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => false)
sub_solver_HiGHS = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
solver_Gurobi = Gurobi.Optimizer
solver_HiGHS = HiGHS.Optimizer

# Original parameters as defined in the problem
alfa_pt = [
    [1.38, 1.67, 2.22, 3.58],
    [2.72, 3.291, 4.381, 7.055],
    [1.76, 2.13, 2.834, 4.565]
]

beta_pt = [
    [85, 102.85, 136.89, 220.46],
    [73, 88.33, 117.56, 189.34],
    [110, 133.1, 177.15, 285.31]
]

sigma_pt = [
    [0.40, 0.48, 0.64, 1.03],
    [0.60, 0.72, 0.96, 1.55],
    [0.50, 0.60, 0.80, 1.29]
]

A_jt_k = [[6.0, 7.26, 9.66, 15.56],
               [20.0, 24.2, 32.21, 51.87],
               [0.0, 0.0, 0.0, 0.0]]

D_jt_k = [[0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0],
               [30.0, 36.3, 48.31, 77.81]]

gamma_jt_k = [[0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0],
                   [26.2, 31.7, 42.19, 67.95]]
phi_jt_k = [[4.0, 4.84, 6.44, 10.37],
                 [9.6, 11.61, 15.46, 24.9],
                 [0.0, 0.0, 0.0, 0.0]]
E_pt_L_original = [1, 10, 10]
E_pt_U_original = [6, 30, 30]
Q_pt_U = [100, 100, 100]

eta_jp = [
    [1.11, 0,    0],
    [0,    1.22, 1.05],
    [0,    0,    0]
]
mu_jp = [
    [0,    0,    0],
    [1,    0,    0],
    [0,    1,    1]
]

scens = 100

using MAT
matfile = matread((@__DIR__)*"/sp_primal_chemical_process_simp.mat")


function interpolate_parameters(T_new::Int)
    # Validate T_new is at least 1
    @assert T_new >= 1 "T_new must be at least 1"

    # Original time indices (1-based)
    original_time = 1:4
    new_time = range(1.0, 4.0, length=T_new)

    # Function to interpolate and round a matrix
    function interpolate_matrix(matrix)
        n_rows = length(matrix)
        n_cols = length(matrix[1])  # Assuming all rows have the same number of columns
        interpolated = Vector{Vector{Float64}}(undef, n_rows)
        for i in 1:n_rows
            # Extract the row values
            row = matrix[i]
            # Create interpolation function
            itp = linear_interpolation(original_time, row, extrapolation_bc=Line())
            # Evaluate at new_time points and round to 3 decimal places
            interpolated_row = [round(itp(t), digits=3) for t in new_time]
            interpolated[i] = interpolated_row
        end
        return interpolated
    end

    # Interpolate each parameter matrix
    interpolated_alfa = interpolate_matrix(alfa_pt)
    interpolated_beta = interpolate_matrix(beta_pt)
    interpolated_sigma = interpolate_matrix(sigma_pt)
    interpolated_A_jt = interpolate_matrix(A_jt_k)
    interpolated_D_jt = interpolate_matrix(D_jt_k)
    interpolated_gamma_jt = interpolate_matrix(gamma_jt_k)
    interpolated_phi_jt = interpolate_matrix(phi_jt_k)

    # Adjust E_pt_L and E_pt_U and round to 3 decimal places
    E_pt_L_new = [round(E_pt_L_original[p] * (4.0 / T_new), digits=3) for p in 1:3]
    E_pt_U_new = [round(min(E_pt_U_original[p], Q_pt_U[p] / T_new), digits=3) for p in 1:3]

    # Ensure E_pt_L_new <= E_pt_U_new
    for p in 1:3
        if E_pt_L_new[p] > E_pt_U_new[p]
            E_pt_L_new[p] = E_pt_U_new[p]
        end
    end

    # Return all interpolated and adjusted parameters
    return (
        alfa_pt=interpolated_alfa,
        beta_pt=interpolated_beta,
        sigma_pt=interpolated_sigma,
        A_jt_k=interpolated_A_jt,
        D_jt_k=interpolated_D_jt,
        gamma_jt_k=interpolated_gamma_jt,
        phi_jt_k=interpolated_phi_jt,
        E_pt_L=E_pt_L_new,
        E_pt_U=E_pt_U_new,
        Q_pt_U=Q_pt_U
    )
end

function generate_random_scenarios(result; num_scenarios=scens, seed=123)
    # Set a seed for reproducibility
    Random.seed!(seed)

    # Define the original matrices
    original_A = deepcopy(result.A_jt_k)
    original_D = deepcopy(result.D_jt_k)
    original_phi = deepcopy(result.phi_jt_k)
    original_gamma = deepcopy(result.gamma_jt_k)

    # Define upper bounds for each parameter
    upper_bounds = Dict(
        :A1 => 20.0,
        :A2 => 70.0,
        :D3 => 100.0,
        :phi1 => 15.0,
        :phi2 => 35.0,
        :gamma3 => 100.0
    )

    # Generate scenarios
    scenarios = []
    probability = round(1.0 / num_scenarios, digits=6)

    for _ in 1:num_scenarios
        # Create deep copies of the original matrices
        A = deepcopy(original_A)
        D = deepcopy(original_D)
        phi = deepcopy(original_phi)
        gamma = deepcopy(original_gamma)

        # Function to generate bounded random values
        function generate_bounded_value(original_value, upper_bound)
            # Generate perturbed value
            perturbed = original_value * (1 + 0.1 * randn())
            # Enforce bounds: 0 ≤ value ≤ upper_bound
            bounded_value = max(0.0, min(perturbed, upper_bound))
            # Round to 3 decimal places
            return round(bounded_value, digits=3)
        end

        # Generate values for each parameter with bounds enforcement
        A[1] = [generate_bounded_value(v, upper_bounds[:A1]) for v in original_A[1]]
        A[2] = [generate_bounded_value(v, upper_bounds[:A2]) for v in original_A[2]]
        D[3] = [generate_bounded_value(v, upper_bounds[:D3]) for v in original_D[3]]
        phi[1] = [generate_bounded_value(v, upper_bounds[:phi1]) for v in original_phi[1]]
        phi[2] = [generate_bounded_value(v, upper_bounds[:phi2]) for v in original_phi[2]]
        gamma[3] = [generate_bounded_value(v, upper_bounds[:gamma3]) for v in original_gamma[3]]

        # Store the scenario with its probability
        push!(scenarios, (
            A = A,
            D = D,
            phi = phi,
            gamma = gamma,
            probability = probability
        ))
    end

    return scenarios
end


using Distributions, Random

function run_MP(time_period = 20, solver = sub_solver_Gurobi; scens = scens, relax = false, multicut = true, fast_math = true)
    # Sets
    P = 1:3  # Processes
    J = 1:3  # Chemicals
    T = 1:time_period  # Time periods

    result = interpolate_parameters(time_period)
    alfa_pt = result.alfa_pt
    beta_pt = result.beta_pt
    E_pt_L = result.E_pt_L
    E_pt_U = result.E_pt_U
    sigma_pt = result.sigma_pt

    # Convert to vectors and filter nonzeros
    nonzero_gamma = vec(result.gamma_jt_k)[result.gamma_jt_k .!= 0.0]
    nonzero_sigma = vec(result.sigma_pt)[result.sigma_pt .!= 0.0]
    nonzero_phi   = vec(result.phi_jt_k)[result.phi_jt_k .!= 0.0]
    nonzero_A     = vec(result.A_jt_k)[result.A_jt_k .!= 0.0]
    nonzero_D     = vec(result.D_jt_k)[result.D_jt_k .!= 0.0]

    # Horizontally stack all
    stacked_data = vcat(nonzero_gamma, nonzero_sigma, nonzero_phi, nonzero_A, nonzero_D)

    # Save to CSV
    writedlm("Files copy/nonzero_values_stacked.csv", stacked_data, ',')

    scenarios = generate_random_scenarios(result, num_scenarios=scens)

    g = OptiGraph()
    g1 = OptiGraph()
    @optinode(g1, master)
    # First-stage variables
    @variable(master, x[p in P, t in T] >= 0)  # Capacity expansion
    @variable(master, y[p in P, t in T], Bin)   # Binary expansion decision
    @variable(master, q[p in P, t in T] >= 0)   # Total capacity

    # First-stage constraints
    @constraint(master, [p in P, t in T], -x[p, t] + E_pt_L[p] * y[p, t] <= 0)
    @constraint(master, [p in P, t in T], x[p, t] - E_pt_U[p] * y[p, t] <= 0)
    @constraint(master, [p in P], q[p, 1] == x[p, 1])
    @constraint(master, [p in P, t=2:time_period], q[p,t] == q[p,t-1] + x[p,t])
    @constraint(master, [p in P, t in T], q[p,t] <= Q_pt_U[p])

    # Objective: Minimize first-stage cost - η (profit)
    @objective(master, Min,
        sum(alfa_pt[p][t] * x[p,t] + beta_pt[p][t] * y[p,t] for p in P, t in T)
    )
    set_to_node_objectives(g1)
    add_subgraph(g, g1)

    function build_subproblem(t, master, g)
        g_sub = OptiGraph()

        @optinode(g_sub, sub)
        @variable(sub, q_copy[P] >= 0)

        add_subgraph(g, g_sub)

        @linkconstraint(g, [p in P], master[:q][p, t] == q_copy[p])
        obj_start = GenericAffExpr{Float64, Plasmo.NodeVariableRef}(0.0)
        set_objective_function(g_sub, obj_start)
        set_optimizer(g_sub, () -> MPBenders.Optimizer())
    end

    function set_optimizer_data(g_sub, idx, scen; fast_math = fast_math)
        opt = MPBenders.get_optimizer(g_sub)
        opt.mp_data.A = matfile["A_list"]
        opt.mp_data.C = matfile["C_list"]
        opt.mp_data.E = matfile["E_list"]
        opt.mp_data.H = matfile["H"]
        opt.mp_data.b = matfile["b_list"]
        opt.mp_data.d = matfile["d_list"]
        opt.mp_data.f = matfile["f_list"]
        opt.mp_data.c = matfile["c"]
        opt.mp_data.fast_math = fast_math

        opt.mp_data.multiplier = fill(1.0, 9, 1)
        opt.mp_data.multiplier[1:3] = matfile["multiplier"][:]
        opt.mp_data.obj_multiplier = 1.0/scens
        #println(sigma_scenarios[1:3, idx, scen])

        n = all_nodes(g_sub)[1]
        opt.mp_data.param_values = zeros(12, 1)
        opt.mp_data.param_values[4]  = scenarios[scen].gamma[3][idx]
        opt.mp_data.param_values[5]  = sigma_pt[1][idx]
        opt.mp_data.param_values[6]  = sigma_pt[2][idx]
        opt.mp_data.param_values[7]  = sigma_pt[3][idx]
        opt.mp_data.param_values[8]  = scenarios[scen].phi[1][idx]
        opt.mp_data.param_values[9]  = scenarios[scen].phi[2][idx]
        opt.mp_data.param_values[10] = scenarios[scen].A[1][idx]
        opt.mp_data.param_values[11] = scenarios[scen].A[2][idx]
        opt.mp_data.param_values[12] = scenarios[scen].D[3][idx]

        opt.mp_data.param_map = Dict(n[:_comp_vars_copy][1] => 1,
                                    n[:_comp_vars_copy][2] => 2,
                                    n[:_comp_vars_copy][3] => 3,
        )
        opt.mp_data.dual_idx = Dict(n[:_comp_vars_copy][1] => [1],
                                    n[:_comp_vars_copy][2] => [2],
                                    n[:_comp_vars_copy][3] => [3]
        )
        opt.mp_data.graph = g_sub
    end

    for s in 1:scens
        for t in T
            build_subproblem(t, master, g)
        end
    end

    set_optimizer(g1, solver)
    #set_optimizer(g1, optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))

    ba = BendersAlgorithm(g, local_subgraphs(g)[1], M = -1e5, max_iters = 50, multicut=multicut, parallelize_benders=false);

    gs = local_subgraphs(g)
    for s in 1:scens
        for t in T
            g_idx = ((s - 1) * T[end] + t + 1)
            set_optimizer_data(gs[g_idx], t, s)
        end
    end

    if relax
        bin_list = []
        avs = all_variables(gs[1])
        for v in avs
            if is_binary(v)
                unset_binary(v)
                set_upper_bound(v, 1)
                set_lower_bound(v, 0)
                push!(bin_list, v)
            end
        end
        t = @elapsed begin
            run_algorithm!(ba)
            ba.best_upper_bound = Inf
            for v in bin_list
                set_binary(v)
            end
            run_algorithm!(ba)
            println(ba.best_upper_bound)
        end
    else
        t = @elapsed run_algorithm!(ba)
        println(ba.best_upper_bound)
    end

    return t, ba
end

function run_Plasmo(time_period = 20, solver = sub_solver_Gurobi, BD::Bool = true; scens = scens, relax = false, multicut = true)

    # Sets
    P = 1:3  # Processes
    J = 1:3  # Chemicals
    T = 1:time_period  # Time periods

    result = interpolate_parameters(time_period)
    alfa_pt = result.alfa_pt
    beta_pt = result.beta_pt
    E_pt_L = result.E_pt_L
    E_pt_U = result.E_pt_U
    sigma_pt = result.sigma_pt
    scenarios = generate_random_scenarios(result, num_scenarios=scens)

    g = OptiGraph()
    g1 = OptiGraph()
    @optinode(g1, master)
    # First-stage variables
    @variable(master, x[p in P, t in T] >= 0)  # Capacity expansion
    @variable(master, y[p in P, t in T], Bin)   # Binary expansion decision
    @variable(master, q[p in P, t in T] >= 0)   # Total capacity

    # First-stage constraints
    @constraint(master, [p in P, t in T], -x[p, t] + E_pt_L[p] * y[p, t] <= 0)
    @constraint(master, [p in P, t in T], x[p, t] - E_pt_U[p] * y[p, t] <= 0)
    @constraint(master, [p in P], q[p, 1] == x[p, 1])
    @constraint(master, [p in P, t=2:time_period], q[p,t] == q[p,t-1] + x[p,t])
    @constraint(master, [p in P, t in T], q[p,t] <= Q_pt_U[p])

    # Objective: Minimize first-stage cost - η (profit)
    @objective(master, Min,
        sum(alfa_pt[p][t] * x[p,t] + beta_pt[p][t] * y[p,t] for p in P, t in T)
    )
    add_subgraph(g, g1)


    function build_subproblem(t, master, g, scen)
        g_sub = OptiGraph()

        #println(sigma_scenarios[1:3, t, scen])
        @optinode(g_sub, sub)
        @variable(sub, w[p in P] >= 0)
        @variable(sub, v[j in 1:2] >= 0)
        @variable(sub, z[j in 3:3] >= 0)

        # Second-stage constraints
        @constraint(sub, [j in 1:2], v[j] <=scenarios[scen].A[j][t])
        @constraint(sub, [j in 3:3], z[j] <= scenarios[scen].D[j][t])
        @constraint(sub, [j in 1:1], v[j] - sum((eta_jp[j][p] - mu_jp[j][p]) * w[p] for p in P ) == 0)
        @constraint(sub, [j in 2:2], v[j] - sum((eta_jp[j][p] - mu_jp[j][p]) * w[p] for p in P ) == 0)
        @constraint(sub, [j in 3:3], - z[j] - sum((eta_jp[j][p] - mu_jp[j][p]) * w[p] for p in P ) == 0)

        # Objective for subproblem t
        @objective(sub, Min,
            -(sum(scenarios[scen].gamma[j][t] * z[j] for j in 3:3) -
            sum(sigma_pt[p][t] * w[p] for p in P) -
            sum(scenarios[scen].phi[j][t] * v[j] for j in 1:2)) / scens
        )

        add_subgraph(g, g_sub)

        @linkconstraint(g, [p in P], w[p] - master[:q][p, t] <= 0)

        ## Fix q[p,t] from master
        #@constraint(sub, [p in P], w[p] - d[p] <= 0)
        #sub[:d_eq_constraint] = @constraint(sub, [p in P], d[p] == q_bar[p,t])
    end

    for s in 1:scens
        for t in T
            build_subproblem(t, master, g, s)
        end
    end
    gs = local_subgraphs(g)
    if BD
        ba = BendersAlgorithm(g, local_subgraphs(g)[1], solver = solver, M = -1e5, max_iters=50, parallelize_benders=false, multicut = multicut)
        if relax
            bin_list = []
            avs = all_variables(gs[1])
            for v in avs
                if is_binary(v)
                    unset_binary(v)
                    set_upper_bound(v, 1)
                    set_lower_bound(v, 0)
                    push!(bin_list, v)
                end
            end
            t = @elapsed begin
                run_algorithm!(ba)
                ba.best_upper_bound = Inf
                for v in bin_list
                    set_binary(v)
                end
                run_algorithm!(ba)
                println(ba.best_upper_bound)
            end
        else
            t = @elapsed run_algorithm!(ba)
            println(ba.best_upper_bound)
        end
        return t, ba
    else
        set_to_node_objectives(g)
        set_optimizer(g, solver)
        t = @elapsed optimize!(g)
        return t

    end
end


sub_solver_Gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => false)
sub_solver_HiGHS = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
solver_Gurobi = Gurobi.Optimizer
solver_HiGHS = HiGHS.Optimizer

run_Plasmo(4, solver_HiGHS, false; scens = 1)
run_Plasmo(4, sub_solver_HiGHS, true; scens = 1)
run_Plasmo(4, sub_solver_HiGHS, true; relax = true, scens = 1)
run_MP(4, sub_solver_HiGHS; relax = true, scens = 1)
run_MP(4, sub_solver_HiGHS; relax = false, scens = 1)




# scens_list = [10, 50, 100, 200, 500]
# time_list = [5, 10, 25, 50]

# t_HiGHS, ba_HiGHS = run_MP(15, sub_solver_HiGHS; relax = false, scens = 500, fast_math = false);
#t_Gurobi, ba_Gurobi = run_MP(5, sub_solver_Gurobi; relax = false, scens = 1500)


t_HiGHS_fast, ba_HiGHS_fast = run_MP(5, sub_solver_Gurobi; relax = false, scens = 500, multicut = true)
# # _HiGHS_BD, ba_HiGHS_BD = run_Plasmo(5, sub_solver_HiGHS; relax = false, scens = 100, multicut =t true)
# # t_Gurobi_BD, ba_Gurobi_BD = run_Plasmo(5, sub_solver_Gurobi; relax = false, scens = 100, multicut = true)

# t_HiGHS_BD, ba_HiGHS_BD = run_Plasmo(15, sub_solver_HiGHS; relax = false, scens = 500, multicut = true)
# t_Gurobi_BD, ba_Gurobi_BD = run_Plasmo(4, sub_solver_Gurobi; relax = false, scens = 500, multicut = true)

# println(ba_HiGHS.ext["timer_optimize"][1] - ba_HiGHS.ext["timer_master"][1])
println(ba_HiGHS_fast.ext["timer_optimize"][1] - ba_HiGHS_fast.ext["timer_master"][1])
# println(ba_HiGHS_BD.ext["timer_optimize"][1] - ba_HiGHS_BD.ext["timer_master"][1])
# println(ba_Gurobi_BD.ext["timer_optimize"][1] - ba_Gurobi_BD.ext["timer_master"][1])
# sgs = local_subgraphs(ba_HiGHS_fast.graph)
# MPBenders.get_optimizer(sgs[2]).mp_data.opt_idx
#=
println(ba_Gurobi.ext["timer_optimize"][1] - ba_Gurobi.ext["timer_master"][1])
println(ba_Gurobi_BD.ext["timer_optimize"][1] - ba_Gurobi_BD.ext["timer_master"][1])

println("HiGHS MP")
println("Total time = $(t_HiGHS), Total FP = $(sum(ba_HiGHS.time_forward_pass)), Master solve = $(ba_HiGHS.ext["timer_master"][1]), optimize solves = $(ba_HiGHS.ext["timer_optimize"][1])")
println("HiGHS BD")
println("Total time = $(t_HiGHS_BD), Total FP = $(sum(ba_HiGHS_BD.time_forward_pass)), Master solve = $(ba_HiGHS_BD.ext["timer_master"]), optimize solves = $(ba_HiGHS_BD.ext["timer_optimize"])")


println("Gurobi MP")
println("Total time = $(t_Gurobi), Total FP = $(sum(ba_Gurobi.time_forward_pass)), Master solve = $(ba_Gurobi.ext["timer_master"]), optimize solves = $(ba_Gurobi.ext["timer_optimize"])")
println("Gurobi BD")
println("Total time = $(t_Gurobi_BD), Total FP = $(sum(ba_Gurobi_BD.time_forward_pass)), Master solve = $(ba_Gurobi_BD.ext["timer_master"]), optimize solves = $(ba_Gurobi_BD.ext["timer_optimize"])")
=#
# function get_primal_sols(optimizer::PlasmoBenders.BendersAlgorithm; idx = 5)
#     graph = optimizer.graph
#     sgs = local_subgraphs(graph)
#     primal_iters = optimizer.primal_iters
#     primal_sols = primal_iters[sgs[idx]]
#     var_list = optimizer.comp_vars[sgs[idx]]
#     best_sols = value(optimizer, var_list)
#     return var_list, primal_sols, best_sols
# end
# comp_vars, sols, best_sols = get_primal_sols(ba_HiGHS_fast)
# sols

function get_primal_sols(optimizer::PlasmoBenders.BendersAlgorithm; idx = 5)
    graph = optimizer.graph
    sgs = local_subgraphs(graph)
    primal_iters = optimizer.primal_iters
    primal_sols = primal_iters[sgs[idx]]
    var_list = optimizer.comp_vars[sgs[idx]]
    best_sols = value(optimizer, var_list)
    return var_list, primal_sols, best_sols
end

# Save all time periods (1 to 4) as CSV
for idx in 2:6
    _, sols_, _ = get_primal_sols(ba_HiGHS_fast; idx=idx)
    df = DataFrame(sols_', :auto)  # Transpose so iteration is along rows
    CSV.write("Files copy/primal_sols_time_$(idx - 1).csv", df)
end