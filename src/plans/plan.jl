"""
    status_summary(e)

Return a string reporting about the current status of `e`,
where `e` is a type from Manopt, e.g. an [`AbstractManoptSolverState`](@ref)s.

This method is similar to `show` but just returns a string.
It might also be more verbose in explaining, or hide internal information.
"""
status_summary(e) = "$(e)"

"""
    set_manopt_parameter!(f, element::Symbol , args...)

For any `f` and a `Symbol` `e` we dispatch on its value so by default, to
set some `args...` in `f` or one of uts sub elements.
"""
function set_manopt_parameter!(f, e::Symbol, args...)
    return set_manopt_parameter!(f, Val(e), args...)
end
function set_manopt_parameter!(f, args...)
    return f
end

"""
    get_manopt_parameter(f, element::Symbol, args...)

Access arbitrary parameters from `f` adressedby a symbol `element`.

For any `f` and a `Symbol` `e` we dispatch on its value so by default, to
get some element from `f` potentially further qulalified by `args...`.

This functions returns `nothing` if `f` does not have the property `element`
"""
function get_manopt_parameter(f, e::Symbol, args...)
    return get_manopt_parameter(f, Val(e), args...)
end
get_manopt_parameter(f, args...) = nothing

"""
    get_manopt_parameter(element::Symbol; default=nothing)

Access global [`Manopt`](@ref) parametersadressed by a symbol `element`.
We first dispatch on the value of `element`.

if the value is not set, `default` is returned

The parameters are queried from the global settings using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl),
so they are persistent within your activated Environment.

# Currenlty used settings

`:Mode`
the mode can be set to `"Tutorial"` to get several hints especially in scenarios, where
the optimisation on manifolds is different from the usual “experience” in
(classicall, Euclidean) optimization.
Any other value has the same effect as not setting it.
"""
function get_manopt_parameter(
    e::Symbol, args...; default=get_manopt_parameter(Val(e), Val(:default))
)
    return @load_preference("$(e)", default)
end
# Handle empty defaults
get_manopt_parameter(e::Symbol, v::Val{:default}) = nothing
get_manopt_parameter(::Val{:Mode}, v::Val{:default}) = ""

"""
    set_manopt_parameter!(element::Symbol, value::Union{String,Bool})

Set global [`Manopt`](@ref) parameters adressed by a symbol `element`.
We first dispatch on the value of `element`.

The parameters are stored to the global settings using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl).

Passing a `value` of `""` deletes an entry from the preferences.
Whenever the `LocalPreferences.toml` is modified, this is also `@info`rmed about.
"""
function set_manopt_parameter!(e::Symbol, value::Union{String,Bool})
    if length(value) == 0
        @delete_preferences!("$(e)")
        v = get_manopt_parameter(e, Val(:default))
        default = isnothing(v) ? "" : (length(v) == 0 ? "" : " ($(get_manopt_parameter))")
        @info("Resetting the `Manopt.jl` parameter $(e) to default$(default).")
    else
        @set_preferences!("$(e)" => value)
        @info("Setting the `Manopt.jl` parameter $(e) to $value.")
    end
end

include("objective.jl")
include("problem.jl")
include("solver_state.jl")

include("debug.jl")
include("record.jl")

include("stopping_criterion.jl")
include("stepsize.jl")
include("cost_plan.jl")
include("gradient_plan.jl")
include("hessian_plan.jl")
include("proximal_plan.jl")
include("subgradient_plan.jl")

include("subsolver_plan.jl")
include("constrained_plan.jl")
include("trust_regions_plan.jl")

include("adabtive_regularization_with_cubics_plan.jl")
include("alternating_gradient_plan.jl")
include("augmented_lagrangian_plan.jl")
include("conjugate_gradient_plan.jl")
include("exact_penalty_method_plan.jl")
include("frank_wolfe_plan.jl")
include("quasi_newton_plan.jl")
include("nonlinear_least_squares_plan.jl")
include("difference_of_convex_plan.jl")
include("Douglas_Rachford_plan.jl")

include("primal_dual_plan.jl")
include("higher_order_primal_dual_plan.jl")

include("stochastic_gradient_plan.jl")

include("embedded_objective.jl")

include("cache.jl")
include("count.jl")
