using KrylovKit

@doc raw"""
    HomogeneousState{P,T} <: AbstractHessianSolverState

A state for the [`homogeneous_descent`](@ref) solver.

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `η1`, `η2`:           (`0.1`, `0.9`) bounds for evaluating the regularization parameter
* `γ1`, `γ2`:           (`0.1`, `2.0`) shrinking and expansion factors for regularization parameter `σ`
* `p`:                  (`rand(M)` the current iterate
* `X`:                  (`zero_vector(M,p)`) the current gradient ``\operatorname{grad}f(p)``
* `s`:                  (`zero_vector(M,p)`) the tangent vector step resulting from minimizing the model
  problem in the tangent space ``\mathcal T_{p} \mathcal M``
* `σ`:                  the current cubic regularization parameter
* `σmin`:               (`1e-7`) lower bound for the cubic regularization parameter
* `ρ_regularization`:   (`1e3`) regularization parameter for computing ρ.
 When approaching convergence ρ may be difficult to compute with numerator and denominator approaching zero.
 Regularizing the ratio lets ρ go to 1 near convergence.
* `evaluation`:         (`AllocatingEvaluation()`) if you provide a
* `retraction_method`:  (`default_retraction_method(M)`) the retraction to use
* `stopping_criterion`: ([`StopAfterIteration`](@ref)`(100)`) a [`StoppingCriterion`](@ref)
* `sub_problem`:        sub problem solved in each iteration
* `sub_state`:          sub state for solving the sub problem, either a solver state if
  the problem is an [`AbstractManoptProblem`](@ref) or an [`AbstractEvaluationType`](@ref) if it is a function,
  where it defaults to [`AllocatingEvaluation`](@ref).

Furthermore the following integral fields are defined

* `q`:                  (`copy(M,p)`) a point for the candidates to evaluate model and ρ
* `H`:                  (`copy(M, p, X)`) the current Hessian, ``\operatorname{Hess}F(p)[⋅]``
* `S`:                  (`copy(M, p, X)`) the current solution from the subsolver
* `ρ`:                  the current regularized ratio of actual improvement and model improvement.
* `ρ_denominator`:      (`one(ρ)`) a value to store the denominator from the computation of ρ
  to allow for a warning or error when this value is non-positive.

# Constructor

    HomogeneousState(M, p=rand(M); X=zero_vector(M, p); kwargs...)

Construct the solver state with all fields stated as keyword arguments.
"""
mutable struct HomogeneousState{
    P,
    T,
    Pr<:Union{AbstractManoptProblem,<:Function},
    St<:Union{AbstractManoptSolverState,<:AbstractEvaluationType},
    TStop<:StoppingCriterion,
    R,
    TRTM<:AbstractRetractionMethod,
} <: AbstractManoptSolverState
    p::P
    X::T
    sub_problem::Pr
    sub_state::St
    q::P
    H::T
    S::T
    σ::R
    ρ::R
    ρ_denominator::R
    ρ_regularization::R
    stop::TStop
    retraction_method::TRTM
    σmin::R
    η1::R
    η2::R
    γ1::R
    γ2::R
end

function HomogeneousState(
    M::AbstractManifold,
    p::P=rand(M),
    X::T=zero_vector(M, p);
    sub_objective=nothing,
    sub_problem::Pr=if isnothing(sub_objective)
        nothing
    else
        DefaultManoptProblem(TangentSpace(M, copy(M, p)), sub_objective)
    end,
    sub_state::St=if sub_problem isa Function
        AllocatingEvaluation()
    else
        HomogeneousLanczosState(TangentSpace(M, copy(M, p)))
    end,
    σ::R=100.0 / sqrt(manifold_dimension(M)),# Had this to initial value of 0.01. However try same as in MATLAB: 100/sqrt(dim(M))
    ρ_regularization::R=1e3,
    stopping_criterion::SC=StopAfterIteration(100),
    retraction_method::RTM=default_retraction_method(M),
    σmin::R=1e-10,
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
) where {
    P,
    T,
    R,
    Pr<:Union{<:AbstractManoptProblem,<:Function,Nothing},
    St<:Union{<:AbstractManoptSolverState,<:AbstractEvaluationType},
    SC<:StoppingCriterion,
    RTM<:AbstractRetractionMethod,
}
    isnothing(sub_problem) && error("No sub_problem provided,")

    return HomogeneousState{P,T,Pr,St,SC,R,RTM}(
        p,
        X,
        sub_problem,
        sub_state,
        copy(M, p),
        copy(M, p, X),
        copy(M, p, X),
        σ,
        one(σ),
        one(σ),
        ρ_regularization,
        stopping_criterion,
        retraction_method,
        σmin,
        η1,
        η2,
        γ1,
        γ2,
    )
end

get_iterate(s::HomogeneousState) = s.p
function set_iterate!(s::HomogeneousState, p)
    s.p = p
    return s
end
get_gradient(s::HomogeneousState) = s.X
function set_gradient!(s::HomogeneousState, X)
    s.X = X
    return s
end

function show(io::IO, homeos::HomogeneousState)
    i = get_count(homeos, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(homeos.stop) ? "Yes" : "No"
    sub = repr(homeos.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
    s = """
    # Solver state for `Manopt.jl`s Homogeneous Descent Method (HDM)
    $Iter
    ## Parameters
    * η1 | η2              : $(homeos.η1) | $(homeos.η2)
    * γ1 | γ2              : $(homeos.γ1) | $(homeos.γ2)
    * σ (σmin)             : $(homeos.σ) ($(homeos.σmin))
    * ρ (ρ_regularization) : $(homeos.ρ) ($(homeos.ρ_regularization))
    * retraction method    : $(homeos.retraction_method)
    * sub solver state     :
        | $(sub)

    ## Stopping criterion

    $(status_summary(homeos.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    homogeneous_descent(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    homogeneous_descent(M, f, grad_f, p=rand(M); kwargs...)
    homogeneous_descent(M, hmo, p=rand(M); kwargs...)

Solve an optimization problem on the manifold `M` by iteratively minimizing

```math
  m_k(X) = f(p_k) + ⟨X, \operatorname{grad} f(p_k)⟩ + \frac{1}{2}⟨X, \operatorname{Hess} f(p_k)[X]⟩ + \frac{σ_k}{3}\lVert X \rVert^3
```

on the tangent space at the current iterate ``p_k``, where ``X ∈ T_{p_k}\mathcal M`` and
``σ_k > 0`` is a regularization parameter.

Let ``X_k`` denote the minimizer of the model ``m_k`` and use the model improvement

```math
  ρ_k = \frac{f(p_k) - f(\operatorname{retr}_{p_k}(X_k))}{m_k(0) - m_k(X_k) + \frac{σ_k}{3}\lVert X_k\rVert^3}.
```

With two thresholds ``η_2 ≥ η_1 > 0``
set ``p_{k+1} = \operatorname{retr}_{p_k}(X_k)`` if ``ρ ≥ η_1``
and reject the candidate otherwise, that is, set ``p_{k+1} = p_k``.

Further update the regularization parameter using factors ``0 < γ_1 < 1 < γ_2``

```math
σ_{k+1} =
\begin{cases}
    \max\{σ_{\min}, γ_1σ_k\} & \text{ if } ρ \geq η_2 &\text{   (the model was very successful)},\\
    σ_k & \text{ if } ρ ∈ [η_1, η_2)&\text{   (the model was successful)},\\
    γ_2σ_k & \text{ if } ρ < η_1&\text{   (the model was unsuccessful)}.
\end{cases}
```

For more details see [AgarwalBoumalBullinsCartis:2020](@cite).

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f`: (optional) the Hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p`:      an initial value ``p ∈ \mathcal M``

For the case that no Hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and Hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

# Keyword arguments

the default values are given in brackets

* `σ`:                      (`100.0 / sqrt(manifold_dimension(M)`) initial regularization parameter
* `σmin`:                   (`1e-10`) minimal regularization value ``σ_{\min}``
* `η1`:                     (`0.1`) lower model success threshold
* `η2`:                     (`0.9`) upper model success threshold
* `γ1`:                     (`0.1`) regularization reduction factor (for the success case)
* `γ2`:                     (`2.0`) regularization increment factor (for the non-success case)
* `evaluation`:             ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `grad_f(M, p)`
  or [`InplaceEvaluation`](@ref) in place, that is of the form `grad_f!(M, X, p)` and analogously for the Hessian.
* `retraction_method`:      (`default_retraction_method(M, typeof(p))`) a retraction to use
* `initial_tangent_vector`: (`zero_vector(M, p)`) initialize any tangent vector data,
* `maxIterLanczos`:         (`200`) a shortcut to set the stopping criterion in the sub solver,
* `ρ_regularization`:       (`1e3`) a regularization to avoid dividing by zero for small values of cost and model
* `stopping_criterion`:     ([`StopAfterIteration`](@ref)`(40) | `[`StopWhenGradientNormLess`](@ref)`(1e-9) | `[`StopWhenAllHomogeneousLanczosVectorsUsed`](@ref)`(maxIterLanczos)`)
* `sub_state`:              [`HomogeneousLanczosState`](@ref)`(M, copy(M, p); maxIterLanczos=maxIterLanczos, σ=σ)
  a state for the subproblem or an [`AbstractEvaluationType`](@ref) if the problem is a function.
* `sub_objective`:          a shortcut to modify the objective of the subproblem used within in the
* `sub_problem`:            [`DefaultManoptProblem`](@ref)`(M, sub_objective)` the problem (or a function) for the sub problem

All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

By default the `debug=` keyword is set to [`DebugIfEntry`](@ref)`(:ρ_denominator, >(0); message="Denominator nonpositive", type=:error)`
to avoid that by rounding errors the denominator in the computation of `ρ` gets nonpositive.
"""
homogeneous_descent(M::AbstractManifold, args...; kwargs...)

function homogeneous_descent(
    M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
) where {TH<:Function}
    return homogeneous_descent(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
function homogeneous_descent(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    Hess_f::THF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF,THF}
    hmo = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    return homogeneous_descent(M, hmo, p; evaluation=evaluation, kwargs...)
end
function homogeneous_descent(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    Hess_f::THF,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF,THF}
    q = [p]
    f_(M, p) = f(M, p[])
    Hess_f_ = Hess_f
    if evaluation isa AllocatingEvaluation
        grad_f_ = (M, p) -> [grad_f(M, p[])]
        Hess_f_ = (M, p, X) -> [Hess_f(M, p[], X[])]
    else
        grad_f_ = (M, X, p) -> (X .= [grad_f(M, p[])])
        Hess_f_ = (M, Y, p, X) -> (Y .= [Hess_f(M, p[], X[])])
    end
    rs = homogeneous_descent(M, f_, grad_f_, Hess_f_, q; evaluation=evaluation, kwargs...)
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function homogeneous_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return homogeneous_descent(M, f, grad_f, rand(M); kwargs...)
end
function homogeneous_descent(
    M::AbstractManifold,
    f::TF,
    grad_f::TdF,
    p;
    evaluation=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {TF,TdF}
    Hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return homogeneous_descent(
        M,
        f,
        grad_f,
        Hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function homogeneous_descent(
    M::AbstractManifold, hmo::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return homogeneous_descent!(M, hmo, q; kwargs...)
end

@doc raw"""
    homogeneous_descent!(M, f, grad_f, Hess_f, p; kwargs...)
    homogeneous_descent!(M, f, grad_f, p; kwargs...)
    homogeneous_descent!(M, hmo, p; kwargs...)

evaluate the Riemannian homogeneous descent solver in place of `p`.

# Input
* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f`: (optional) the Hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `p`:      an initial value ``p  ∈  \mathcal M``

For the case that no Hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

the cost `f` and its gradient and Hessian might also be provided as a [`ManifoldHessianObjective`](@ref)

for more details and all options, see [`homogeneous_descent`](@ref).
"""
homogeneous_descent!(M::AbstractManifold, args...; kwargs...)
function homogeneous_descent!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return homogeneous_descent!(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function homogeneous_descent!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TH<:Function}
    hmo = ManifoldHessianObjective(f, grad_f, Hess_f; evaluation=evaluation)
    return homogeneous_descent!(M, hmo, p; evaluation=evaluation, kwargs...)
end

# only defined function, all others are decorators to override the default
function homogeneous_descent!(
    M::AbstractManifold,
    hmo::O,
    p=rand(M);
    debug=DebugIfEntry(
        :ρ_denominator, >(-1e-8); message="denominator nonpositive", type=:error
    ),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    initial_tangent_vector::T=zero_vector(M, p),
    maxIterLanczos=min(300, manifold_dimension(M)),
    objective_type=:Riemannian,
    ρ_regularization::R=1e3,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    σmin::R=1e-10,
    σ::R=100.0 / sqrt(manifold_dimension(M)),
    η1::R=0.1,
    η2::R=0.9,
    γ1::R=0.1,
    γ2::R=2.0,
    θ::R=0.5,
    sub_kwargs=(;),
    sub_stopping_criterion::StoppingCriterion=StopAfterIteration(maxIterLanczos) |
                                              StopWhenFirstOrderProgress(θ),
    sub_state::Union{<:AbstractManoptSolverState,<:AbstractEvaluationType}=decorate_state!(
        HomogeneousLanczosState(
            TangentSpace(M, copy(M, p));
            maxIterLanczos=maxIterLanczos,
            σ=σ,
            θ=θ,
            stopping_criterion=sub_stopping_criterion,
            sub_kwargs...,
        );
        sub_kwargs,
    ),
    sub_objective=nothing,
    sub_problem=nothing,
    stopping_criterion::StoppingCriterion=if sub_state isa HomogeneousLanczosState
        StopAfterIteration(140) |
        StopWhenGradientNormLess(1e-9) |
        StopWhenAllHomogeneousLanczosVectorsUsed(maxIterLanczos - 1)
    else
        StopAfterIteration(140) | StopWhenGradientNormLess(1e-9)
    end,
    kwargs...,
) where {T,R,O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    dhmo = decorate_objective!(M, hmo; objective_type=objective_type, kwargs...)
    if isnothing(sub_objective)
        sub_objective = decorate_objective!(
            M, HomogeneousDescentModelObjective(dhmo, σ); sub_kwargs...
        )
    end
    if isnothing(sub_problem)
        sub_problem = DefaultManoptProblem(TangentSpace(M, copy(M, p)), sub_objective)
    end
    X = copy(M, p, initial_tangent_vector)
    dmp = DefaultManoptProblem(M, dhmo)
    homeos = HomogeneousState(
        M,
        p,
        X;
        sub_state=sub_state,
        sub_problem=sub_problem,
        σ=σ,
        ρ_regularization=ρ_regularization,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        σmin=σmin,
        η1=η1,
        η2=η2,
        γ1=γ1,
        γ2=γ2,
    )
    dhomeos = decorate_state!(homeos; debug, kwargs...)
    solve!(dmp, dhomeos)
    return get_solver_return(get_objective(dmp), dhomeos)
end

function initialize_solver!(dmp::AbstractManoptProblem, homeos::HomogeneousState)
    get_gradient!(dmp, homeos.X, homeos.p)
    return homeos
end
function step_solver!(dmp::AbstractManoptProblem, homeos::HomogeneousState, i)
    M = get_manifold(dmp)
    hmo = get_objective(dmp)
    # Update sub state
    # Set point also in the sub problem (eventually the tangent space)

    """
        get_gradient(M::AbstractManifold, mgo::AbstractManifoldGradientObjective{T}, p)
        get_gradient!(M::AbstractManifold, X, mgo::AbstractManifoldGradientObjective{T}, p)

        evaluate the gradient of a [`AbstractManifoldGradientObjective{T}`](@ref) `mgo` at `p`.

        The evaluation is done in place of `X` for the `!`-variant.
        The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
        When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
        memory for the result is allocated.

        Note that the order of parameters follows the philosophy of `Manifolds.jl`, namely that
        even for the mutating variant, the manifold is the first parameter and the (in-place) tangent
        vector `X` comes second.
    """
    get_gradient!(M, homeos.X, hmo, homeos.p)
    # Update base point in manifold
    set_manopt_parameter!(homeos.sub_problem, :Manifold, :p, copy(M, homeos.p))
    set_manopt_parameter!(homeos.sub_problem, :Objective, :σ, homeos.σ)
    set_iterate!(homeos.sub_state, M, copy(M, homeos.p, homeos.X))
    set_manopt_parameter!(homeos.sub_state, :σ, homeos.σ)
    # Solve the `sub_problem` via dispatch depending on type
    θ, info = solve_homogeneous_subproblem!(
        M, homeos.S, homeos.sub_problem, homeos.sub_state, homeos.X, homeos.p
    )

    # Compute ρ
    α = α₀ = 5.0
    kₗ = 0
    ρ_num = 0.0
    ρ_reg = 0.0
    cost = 0.0
    while kₗ < 20
        s = homeos.S .* α
        retract!(M, homeos.q, homeos.p, s, homeos.retraction_method)
        cost = get_cost(M, hmo, homeos.p)
        ρ_num = cost - get_cost(M, hmo, homeos.q)
        ρ_vec = homeos.X + 0.5 * get_hessian(M, hmo, homeos.p, s)
        ρ_den = -inner(M, homeos.p, s, ρ_vec)
        ρ_reg = homeos.ρ_regularization * eps(Float64) * max(abs(cost), 1)
        # homeos.ρ_denominator = ρ_den + ρ_reg # <= 0 -> the default debug kicks in
        homeos.ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)
        # Update iterate
        if (ρ_num > 0) && (homeos.ρ >= homeos.η1)
            copyto!(M, homeos.p, homeos.q)
            get_gradient!(dmp, homeos.X, homeos.p) # only compute gradient when updating the point
            break
        else
            kₗ += 1
            α /= 1.5
        end
    end
    @printf(
        "f:%+.4f, ρ:%.1f, α:%.1e, θ:%+.1e, k:%2d, Δ:%+.1e/%+.1e, |g|:%.1e\n",
        cost,
        homeos.ρ,
        α,
        θ,
        info.numops,
        ρ_num,
        ρ_reg,
        (norm(homeos.X))
    )

    # Update iterate

    return homeos
end

function get_homogeneous_hessian(M, hmo, X, p, ξ; δ=0)
    """
    Y = get_objective_Hessian(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_Hessian!(M, Y, amso::AbstractManifoldSubObjective, p, X)

    Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
    """
    v = ξ[1:(end - 1)]
    t = ξ[end]
    V = reshape(v, size(X)...)
    HV = get_objective_hessian(M, hmo, p, V)
    # vectorized
    hv = HV[:]
    g = X[:]
    return [hv + t * g; g'v + t * δ]
end
#######################################################################
# A temporary approach to compute the subproblem
# @note: cz 2024-04-05
#######################################################################
function solve_homogeneous_subproblem!(
    M, s, problem::P, state::S, X, p
) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
    hmo = get_objective(problem)
    F(ξ) = get_homogeneous_hessian(M, hmo, X, p, ξ)
    n = length(X)
    # compute eigenvalue
    vals, vecs, info = KrylovKit.eigsolve(
        F, n + 1, 1, :SR, Float64; tol=1e-5, issymmetric=true, eager=true
    )

    ξ = vecs[1]
    q = ξ[1:(end - 1)] / ξ[end]
    Q = reshape(q, size(X)...)

    # copyto!(M::AbstractManifold, Y, p, X).
    # Copy the value(s) from `X` to `Y`, where both are tangent vectors from the tangent space at
    # `p` on the [`AbstractManifold`](@ref) `M`.
    # This function defaults to calling `copyto!(Y, X)`, but it might be useful to overwrite the
    # function at the level, where also information from `p` and `M` can be accessed.
    copyto!(M, s, p, Q)
    return -vals[1], info
end
#######################################################################
# inexact Riemannian Lanczos for homogeneous model,
# using self coded Lanczos, see homogeneous_lanczos.jl
# @note: cz 2024-04-05
#######################################################################
# Dispatch on different forms of `sub_solvers`
# function solve_homogeneous_subproblem!(
#     M, s, problem::P, state::S, p
# ) where {P<:AbstractManoptProblem,S<:AbstractManoptSolverState}
#     solve!(problem, state)
#     copyto!(M, s, p, get_solver_result(state))
#     return s
# end
#######################################################################
# not implemented yet
# function solve_homogeneous_subproblem!(
#     M, s, problem::P, ::AllocatingEvaluation, p
# ) where {P<:Function}
#     copyto!(M, s, p, problem(M, p))
#     return s
# end
# function solve_homogeneous_subproblem!(
#     M, s, problem!::P, ::InplaceEvaluation, p
# ) where {P<:Function}
#     problem!(M, s, p)
#     return s
# end
