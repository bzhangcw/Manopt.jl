
#
# HomogeneousLanczos sub solver
#
@doc raw"""
    HomogeneousLanczosState{P,T,SC,B,I,R,TM,V,Y} <: AbstractManoptSolverState

Solve the adaptive regularized subproblem with a HomogeneousLanczos iteration

# Fields

* `stop`:            the stopping criterion
* `σ`:               the current regularization parameter
* `X`:               the Iterate
* `HomogeneousLanczos_vectors`: the obtained HomogeneousLanczos vectors
* `tridig_matrix`:   the tridiagonal coefficient matrix T
* `coefficients`:    the coefficients ``y_1,...y_k`` that determine the solution
* `Hp`:              a temporary vector containing the evaluation of the Hessian
* `Hp_residual`:     a temporary vector containing the residual to the Hessian
* `S`:               the current obtained / approximated solution
"""
mutable struct HomogeneousLanczosState{T,R,SC,SCN,B,TM,C} <: AbstractManoptSolverState
    X::T
    σ::R
    stop::SC
    stop_newton::SCN
    HomogeneousLanczos_vectors::B # ``q_i``
    tridig_matrix::TM  # `T``
    coefficients::C     # `y``
    Hp::T              # `Hess_f`` A temporary vector for evaluations of the Hessian
    Hp_residual::T     # A residual vector
    S::T               # store the tangent vector that solves the minimization problem
end
function HomogeneousLanczosState(
    TpM::TangentSpace;
    X::T=zero_vector(TpM.manifold, TpM.point),
    maxIterLanczos=200,
    θ=0.5,
    stopping_criterion::SC=StopAfterIteration(maxIterLanczos) |
                           StopWhenFirstOrderProgress(θ),
    stopping_criterion_newtown::SCN=StopAfterIteration(200),
    σ::R=10.0,
) where {T,SC<:StoppingCriterion,SCN<:StoppingCriterion,R}
    tridig = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    coeffs = zeros(maxIterLanczos)
    HomogeneousLanczos_vectors = typeof(X)[]
    return HomogeneousLanczosState{
        T,R,SC,SCN,typeof(HomogeneousLanczos_vectors),typeof(tridig),typeof(coeffs)
    }(
        X,
        σ,
        stopping_criterion,
        stopping_criterion_newtown,
        HomogeneousLanczos_vectors,
        tridig,
        coeffs,
        copy(TpM, X),
        copy(TpM, X),
        copy(TpM, X),
    )
end
function get_solver_result(ls::HomogeneousLanczosState)
    return ls.S
end
function set_iterate!(ls::HomogeneousLanczosState, M, X)
    ls.X .= X
    return ls
end
function set_manopt_parameter!(ls::HomogeneousLanczosState, ::Val{:σ}, σ)
    ls.σ = σ
    return ls
end

function show(io::IO, ls::HomogeneousLanczosState)
    i = get_count(ls, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(ls.stop) ? "Yes" : "No"
    vectors = length(ls.HomogeneousLanczos_vectors)
    s = """
    # Solver state for `Manopt.jl`s HomogeneousLanczos Iteration
    $Iter
    ## Parameters
    * σ                         : $(ls.σ)
    * # of HomogeneousLanczos vectors used : $(vectors)

    ## Stopping criteria
    (a) For the HomogeneousLanczos Iteration
    $(status_summary(ls.stop))
    (b) For the Newton sub solver
    $(status_summary(ls.stop_newton))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
# The HomogeneousLanczos Subsolver implementation
#
function initialize_solver!(
    dmp::AbstractManoptProblem{<:TangentSpace}, ls::HomogeneousLanczosState
)
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = TpM.point
    maxIterLanczos = size(ls.tridig_matrix, 1)
    ls.tridig_matrix = spdiagm(maxIterLanczos, maxIterLanczos, [0.0])
    ls.coefficients = zeros(maxIterLanczos)
    for X in ls.HomogeneousLanczos_vectors
        zero_vector!(M, X, p)
    end
    zero_vector!(M, ls.Hp, p)
    zero_vector!(M, ls.Hp_residual, p)
    return ls
end

function step_solver!(
    dmp::AbstractManoptProblem{<:TangentSpace}, ls::HomogeneousLanczosState, i
)
    TpM = get_manifold(dmp)
    p = TpM.point
    M = base_manifold(TpM)
    hmo = get_objective(dmp)
    if i == 1 #the first is known directly
        nX = norm(M, p, ls.X)
        if length(ls.HomogeneousLanczos_vectors) == 0
            push!(ls.HomogeneousLanczos_vectors, ls.X ./ nX)
        else
            copyto!(M, ls.HomogeneousLanczos_vectors[1], p, ls.X ./ nX)
        end
        get_objective_hessian!(M, ls.Hp, hmo, p, ls.HomogeneousLanczos_vectors[1])
        α = inner(M, p, ls.HomogeneousLanczos_vectors[1], ls.Hp)
        # This is also the first coefficient in the tridiagonal matrix
        ls.tridig_matrix[1, 1] = α
        ls.Hp_residual .= ls.Hp - α * ls.HomogeneousLanczos_vectors[1]
        #this is the minimiser of one dimensional model
        ls.coefficients[1] = (α - sqrt(α^2 + 4 * ls.σ * nX)) / (2 * ls.σ)
    else # `i > 1`
        β = norm(M, p, ls.Hp_residual)
        if β > 1e-12 # Obtained new orthogonal HomogeneousLanczos long enough with respect to numerical stability
            if length(ls.HomogeneousLanczos_vectors) < i
                push!(ls.HomogeneousLanczos_vectors, ls.Hp_residual ./ β)
            else
                copyto!(M, ls.HomogeneousLanczos_vectors[i], p, ls.Hp_residual ./ β)
            end
        else # Generate new random vector and
            # modified Gram Schmidt of new vector with respect to Q
            rand!(M, ls.Hp_residual; vector_at=p)
            for k in 1:(i-1)
                ls.Hp_residual .=
                    ls.Hp_residual -
                    inner(M, p, ls.HomogeneousLanczos_vectors[k], ls.Hp_residual) *
                    ls.HomogeneousLanczos_vectors[k]
            end
            if length(ls.HomogeneousLanczos_vectors) < i
                push!(
                    ls.HomogeneousLanczos_vectors,
                    ls.Hp_residual ./ norm(M, p, ls.Hp_residual),
                )
            else
                copyto!(
                    M,
                    ls.HomogeneousLanczos_vectors[i],
                    p,
                    ls.Hp_residual ./ norm(M, p, ls.Hp_residual),
                )
            end
        end
        # Update Hessian and residual
        get_objective_hessian!(M, ls.Hp, hmo, p, ls.HomogeneousLanczos_vectors[i])
        ls.Hp_residual .= ls.Hp - β * ls.HomogeneousLanczos_vectors[i-1]
        α = inner(M, p, ls.Hp_residual, ls.HomogeneousLanczos_vectors[i])
        ls.Hp_residual .= ls.Hp_residual - α * ls.HomogeneousLanczos_vectors[i]
        # Update tridiagonal matrix
        ls.tridig_matrix[i, i] = α
        ls.tridig_matrix[i-1, i] = β
        ls.tridig_matrix[i, i-1] = β
        min_cubic_Newton!(dmp, ls, i)
    end
    copyto!(
        M, ls.S, p, sum(ls.HomogeneousLanczos_vectors[k] * ls.coefficients[k] for k in 1:i)
    )
    return ls
end
#
# Solve HomogeneousLanczos sub problem
#
function min_cubic_Newton!(
    mp::AbstractManoptProblem{<:TangentSpace}, ls::HomogeneousLanczosState, i
)
    TpM = get_manifold(mp)
    p = TpM.point
    M = base_manifold(TpM)
    tol = 1e-16

    gvec = zeros(i)
    gvec[1] = norm(M, p, ls.X)
    λ = opnorm(Array(@view ls.tridig_matrix[1:i, 1:i])) + 2
    T_λ = @view(ls.tridig_matrix[1:i, 1:i]) + λ * I

    λ_min = eigmin(Array(@view ls.tridig_matrix[1:i, 1:i]))
    lower_barrier = max(0, -λ_min)
    k = 0
    y = zeros(i)
    while !ls.stop_newton(mp, ls, k)
        k += 1
        y = -(T_λ \ gvec)
        ynorm = norm(y, 2)
        ϕ = 1 / ynorm - ls.σ / λ # when ϕ is "zero" then y is the solution.
        (abs(ϕ) < tol * ynorm) && break
        # compute the Newton step
        ψ = ynorm^2
        Δy = -(T_λ) \ y
        ψ_prime = 2 * dot(y, Δy)
        # Quadratic polynomial coefficients
        p0 = 2 * ls.σ * ψ^(1.5)
        p1 = -2 * ψ - λ * ψ_prime
        p2 = ψ_prime
        #Polynomial roots
        r1 = (-p1 + sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)
        r2 = (-p1 - sqrt(p1^2 - 4 * p2 * p0)) / (2 * p2)

        Δλ = max(r1, r2) - λ

        #instead of jumping past the lower barrier for λ,
        # jump to midpoint between current and lower λ.
        (λ + Δλ <= lower_barrier) && (Δλ = -0.5 * (λ - lower_barrier))
        #if the steps are too small, exit
        (abs(Δλ) <= eps(λ)) && break
        T_λ = T_λ + Δλ * I
        λ = λ + Δλ
    end
    ls.coefficients[1:i] .= y
    return ls.coefficients
end

#
# Stopping Criteria
#
@doc raw"""
    StopWhenFirstOrderProgress <: StoppingCriterion

A stopping criterion related to the Riemannian adaptive regularization with cubics (ARC)
solver indicating that the model function at the current (outer) iterate,

```math
    m(X) = f(p) + <X, \operatorname{grad}f(p)>
      + \frac{1}{2} <X, \operatorname{Hess} f(p)[X]> +  \frac{σ}{3} \lVert X \rVert^3,
```

defined on the tangent space ``T_{p}\mathcal M`` fulfills at the current iterate ``X_k`` that

```math
m(X_k) \leq m(0)
\quad\text{ and }\quad
\lVert \operatorname{grad} m(X_k) \rVert ≤ θ \lVert X_k \rVert^2
```

# Fields

* `θ`:      the factor ``θ`` in the second condition
* `reason`: a String indicating the reason if the criterion indicated to stop
"""
# mutable struct StopWhenFirstOrderProgress <: StoppingCriterion
#     θ::Float64 #θ
#     reason::String
#     StopWhenFirstOrderProgress(θ::Float64) = new(θ, "")
# end
function (c::StopWhenFirstOrderProgress)(
    dmp::AbstractManoptProblem{<:TangentSpace}, ls::HomogeneousLanczosState, i::Int
)
    if (i == 0)
        c.reason = ""
        return false
    end
    #Update Gradient
    TpM = get_manifold(dmp)
    p = TpM.point
    M = base_manifold(TpM)
    nX = norm(M, p, get_gradient(dmp, p))
    y = @view(ls.coefficients[1:(i-1)])
    Ty = @view(ls.tridig_matrix[1:i, 1:(i-1)]) * y
    ny = norm(y)
    model_grad_norm = norm(nX .* [1, zeros(i - 1)...] + Ty + ls.σ * ny * [y..., 0])
    prog = (model_grad_norm <= c.θ * ny^2)
    prog && (c.reason = "The algorithm has reduced the model grad norm by a factor $(c.θ).")
    return prog
end


@doc raw"""
    StopWhenAllHomogeneousLanczosVectorsUsed <: StoppingCriterion

When an inner iteration has used up all HomogeneousLanczos vectors, then this stopping criterion is
a fallback / security stopping criterion to not access a non-existing field
in the array allocated for vectors.

Note that this stopping criterion (for now) is only implemented for the case that an
[`HomogeneousState`](@ref) when using a [`HomogeneousLanczosState`](@ref) subsolver

# Fields

* `maxHomogeneousLanczosVectors`: maximal number of HomogeneousLanczos vectors
* `reason`:            a String indicating the reason if the criterion indicated to stop

# Constructor

    StopWhenAllHomogeneousLanczosVectorsUsed(maxLancosVectors::Int)

"""
mutable struct StopWhenAllHomogeneousLanczosVectorsUsed <: StoppingCriterion
    maxHomogeneousLanczosVectors::Int
    reason::String
    function StopWhenAllHomogeneousLanczosVectorsUsed(maxHomogeneousLanczosVectors::Int)
        return new(maxHomogeneousLanczosVectors, "")
    end
end
function (c::StopWhenAllHomogeneousLanczosVectorsUsed)(
    ::AbstractManoptProblem,
    arcs::HomogeneousState{P,T,Pr,<:HomogeneousLanczosState},
    i::Int,
) where {P,T,Pr}
    (i == 0) && (c.reason = "") # reset on init
    if (i > 0) &&
       length(arcs.sub_state.HomogeneousLanczos_vectors) == c.maxHomogeneousLanczosVectors
        c.reason = "The algorithm used all ($(c.maxHomogeneousLanczosVectors)) preallocated HomogeneousLanczos vectors and may have stagnated.\n Consider increasing this value.\n"
        return true
    end
    return false
end
function status_summary(c::StopWhenAllHomogeneousLanczosVectorsUsed)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "All HomogeneousLanczos vectors ($(c.maxHomogeneousLanczosVectors)) used:\t$s"
end
indicates_convergence(c::StopWhenAllHomogeneousLanczosVectorsUsed) = false
function show(io::IO, c::StopWhenAllHomogeneousLanczosVectorsUsed)
    return print(
        io,
        "StopWhenAllHomogeneousLanczosVectorsUsed($(repr(c.maxHomogeneousLanczosVectors)))\n    $(status_summary(c))",
    )
end
