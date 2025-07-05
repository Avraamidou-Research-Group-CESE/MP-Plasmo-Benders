

module MPBenders

using JuMP, Plasmo

const MOIU = MOI.Utilities

include((@__DIR__)*"/mp_struct_primal.jl")

const _PARAMETER_OFFSET = 0x00f0000000000000

_is_parameter(x::MOI.VariableIndex) = x.value >= _PARAMETER_OFFSET
_is_parameter(term::MOI.ScalarAffineTerm) = _is_parameter(term.variable)
_is_parameter(term::MOI.ScalarQuadraticTerm) = _is_parameter(term.variable_1) || _is_parameter(term.variable_2)

"""
    Optimizer()

Create a new MPBenders optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    solver#::Union{Nothing,MadNLP.MadNLPSolver}
    result#::Union{Nothing,MadNLP.MadNLPExecutionStats{Float64}}

    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{Symbol,Any}
    sense::MOI.OptimizationSense

    variables::MOI.Utilities.VariablesContainer{Float64}
    list_of_variable_indices::Vector{MOI.VariableIndex}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    mult_x_L::Vector{Union{Nothing,Float64}}
    mult_x_U::Vector{Union{Nothing,Float64}}

    mp_data::MPData
    timer::Vector
end

function Optimizer(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    return Optimizer(
        nothing,
        nothing,
        "",
        false,
        false,
        option_dict,
        MOI.FEASIBILITY_SENSE,
        #Dict{MOI.VariableIndex,Float64}(),
        MOI.Utilities.VariablesContainer{Float64}(),
        MOI.VariableIndex[],
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[],
        Union{Nothing,Float64}[],
        MPData(),
        Any[]
    )
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
}


function MOI.empty!(model::Optimizer)
    model.solver = nothing
    model.result = MOI.OPTIMAL
    model.invalid_model = false
    model.sense = MOI.FEASIBILITY_SENSE
    MOI.empty!(model.variables)
    empty!(model.list_of_variable_indices)
    empty!(model.variable_primal_start)
    empty!(model.mult_x_L)
    empty!(model.mult_x_U)
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.variable_primal_start) &&
           isempty(model.mult_x_L) &&
           isempty(model.mult_x_U) &&
           model.sense == MOI.FEASIBILITY_SENSE
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end

### MOI.Name
MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("max_cpu_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, :max_cpu_time)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, :max_cpu_time, nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[Symbol(p.name)] = value
    # No need to reset model.solver because this gets handled in optimize!.
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawParameter with name $(p.name) is not set.")
    end
    return model.options[p.name]
end

### Variables

"""
    column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    push!(model.variable_primal_start, nothing)
    push!(model.mult_x_L, nothing)
    push!(model.mult_x_U, nothing)
    model.solver = nothing
    x = MOI.add_variable(model.variables)
    push!(model.list_of_variable_indices, x)
    return x
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    if _is_parameter(x)
        return haskey(model.parameters, x)
    end
    return MOI.is_valid(model.variables, x)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return model.list_of_variable_indices
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.list_of_variable_indices)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.is_valid(model.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(model.variables, attr, c)
end

function MOI.add_constraint(model::Optimizer, x::MOI.VariableIndex, set::_SETS)
    index = MOI.add_constraint(model.variables, x, set)
    model.solver = nothing
    return index
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    MOI.set(model.variables, MOI.ConstraintSet(), ci, set)
    model.solver = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.delete(model.variables, ci)
    model.solver = nothing
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    return MOI.is_valid(model.qp_data, ci)
end

function MOI.add_constraint(model::Optimizer, func::_FUNCTIONS, set::_SETS)
    #println(func)
    #println(set)
    #ERROR
    #index = MOI.add_constraint(model.qp_data, func, set)
    model.solver = nothing
    return MOI.ConstraintIndex{typeof(func),typeof(set)}(1)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.ConstraintFunction,
        MOI.ConstraintSet,
        MOI.ConstraintDualStart,
    },
    c::MOI.ConstraintIndex{F,S},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr, c)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.set(model.qp_data, MOI.ConstraintSet(), ci, set)
    model.solver = nothing
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
    value::Union{Real,Nothing},
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.throw_if_not_valid(model, ci)
    MOI.set(model.qp_data, attr, ci, value)
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    if _is_parameter(vi)
        return  # Do nothing
    end
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[column(vi)] = value
    # No need to reset model.solver, because this gets handled in optimize!.
    return
end

### MOI.ConstraintDualStart

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,S}},
) where {S<:_SETS}
    return true
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    model.solver = nothing
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

### ObjectiveFunction

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,<:_FUNCTIONS}},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {F<:Union{MOI.VariableIndex,<:_FUNCTIONS}}
    model.solver = nothing
    return
end

function MOI.optimize!(model::Optimizer)

    t = @elapsed begin
    run_MP!(model.mp_data)
    end
    g = model.mp_data.graph

    obj_sol = model.mp_data.obj_val[1]
    new_obj_func = GenericAffExpr{Float64, Plasmo.NodeVariableRef}(obj_sol)
    set_objective_function(model.mp_data.graph, new_obj_func)

    # Instantiate MadNLP.
    model.result = MOI.OPTIMAL

    push!(model.timer, t)
    return
end

const _STATUS_CODES = Dict(
    MOI.OPTIMAL => MOI.OPTIMAL
)

### MOI.TerminationStatus
function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.invalid_model
        return MOI.INVALID_MODEL
    end
    return MOI.OPTIMAL
end

### MOI.PrimalStatus
function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = model.result.status
    return MOI.OPTIMAL
end

### MOI.DualStatus
function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    return MOI.OPTIMAL
end

### MOI.ObjectiveValue
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.mp_data.obj_val[1]
end

### MOI.VariablePrimal
function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return 0.
end

### MOI.ConstraintDual

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)

    con = model.mp_data.graph.backend.graph_to_element_map.con_map[ci]
    con_obj = JuMP.constraint_object(con)
    var = con_obj.func

    dual_val = get_dual(model.mp_data, var)
    return dual_val
end


function MOI.check_result_index_bounds(model::Optimizer, attr)
    return
end

function get_optimizer(g::OptiGraph)
    return g.backend.moi_backend.optimizer.model
end

end
