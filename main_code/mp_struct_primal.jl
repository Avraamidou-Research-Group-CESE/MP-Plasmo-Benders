using LinearAlgebra
using LoopVectorization, StaticArrays
using Base.Threads

mutable struct MPData
    A::Array
    C::Array
    E::Array
    H::Array
    b::Array
    d::Array
    f::Array
    c::Array

    multiplier::Array

    obj_multiplier::Float64

    param_values::Array # needs to be N x 1 in shape
    param_map::Dict # this needs to map variables to the index in the `param_values` vector

    dual_sols::Array
    primal_sols::Array
    dual_idx::Dict

    obj_val::Float64
    graph::OptiGraph
    timer::Vector
    fast_math::Bool
    opt_idx::Vector

    #TODO:
    # update optimize call so that it updates all the MPData with any fixed variable refs
    # define interface to set data on an OptiGraph (tests if the optimizer is dype MPData), and then adds this
    # update objective value inside the optimizer as well, remove "scale" command
end


function fast_loop!(data, A, b, C, d, E, f, y, opt_idx)
    @inbounds @fastmath begin
        # Find first index where all constraints are satisfied (E[i]*y <= f[i] + ε)
        idx = findfirst(i -> begin
            Ey = E[i] * y  # Matrix-vector product for this constraint set
            all(Ey .<= f[i] .+ 1e-9)
        end, eachindex(E))

        if idx === nothing
            error("No feasible solution found where E[i]*y - f[i] <= 0")
        end

        # Optimized matrix operations using LoopVectorization
        @tturbo data.primal_sols = A[idx,:,:] * y .+ b[idx,:]
        @tturbo data.dual_sols = (C[idx, :, :] * y .+ d[idx, :]) .* data.multiplier

        push!(opt_idx, idx)
    end
end





function MPData(; kwargs...)
    option_dict = Dict{Symbol, Any}()
    for (name, value) in kwargs
        option_dict[name] = value
    end
    return MPData(
        zeros(0,0,0),
        zeros(0,0,0),
        zeros(0,0,0),
        zeros(0,0,0),
        zeros(0,0),
        zeros(0,0),
        zeros(0,0),
        zeros(0,0),
        zeros(0,0),
        1.,
        zeros(0,0),
        Dict(),
        zeros(0,0),
        zeros(0,0),
        Dict(),
        0.,
        OptiGraph(),
        Any[],
        true,
        Any[]
    )
end


function run_MP!(data::MPData)
    param_map = data.param_map
    multiplier = data.multiplier
    y = data.param_values
    for var in keys(param_map)
        if is_fixed(var)
            y[param_map[var]] = fix_value(var)
        end
    end

    A = data.A
    b = data.b
    C = data.C
    d = data.d
    E = data.E
    f = data.f
    H = data.H
    c = data.c

    opt_idx = data.opt_idx

    t = @elapsed begin
    if data.fast_math
        fast_loop!(data, A, b, C, d, E, f, y, opt_idx)
    else
        for i in 1:size(A, 1)
            #println(E[i, :, :] * y .- f[i, :, :])
            if all(E[i][:, :] * y .- f[i][:, :] .<= 1e-5)

                data.primal_sols = A[i, :, :] * y .+ b[i, :]
                data.dual_sols = C[i, :, :] * y .+ d[i, :]
                data.dual_sols .*= data.multiplier
                push!(opt_idx, i)
                break
            end
            if i == size(A, 1)
                error("Loop concluded without Ey - f <= 0")
            end
        end
    end
    end
    push!(data.timer, t)

    #data.dual_sols .= x
    x = data.primal_sols

    #println("y = ", size(y))
    #println("x = ", size(x))
    #println("H = ", size(H))

    obj_val = 1 * (y' * H * x + c' * x)
    data.obj_val = obj_val[1] * data.obj_multiplier
    data.dual_sols .*= data.obj_multiplier

    return nothing
end

function get_dual(data::MPData, var_ref)
    if !(var_ref in keys(data.param_map)) || !(var_ref in keys(data.dual_idx))
        error("$var_ref is not in the parameter map")
    end

    dual_idx = data.dual_idx[var_ref]

    return - data.dual_sols[dual_idx[1]]# - data.dual_sols[dual_idx[2]]
end

function get_objective_value(data::MPData)

end

_MPoptions_fields = [
    :A,
    :C,
    :E,
    :H,
    :b,
    :d,
    :f,
    :c,
    :param_values,
    :param_map
]
#=
for field in _MPoptions_fields
    @eval begin
        method = Symbol("set_MPdata_", field)
        $method(data::MPData, val) = data.options.$field = val
    end
end
=#
