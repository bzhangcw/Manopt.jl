using Manifolds, ManifoldsBase, Manopt, Test, Random
using LinearAlgebra: I, tr, Symmetric

Random.seed!(42)
n = 8
k = 3
A = Symmetric(randn(n, n))
M = Grassmann(n, k)

f(M, p) = -0.5 * tr(p' * A * p)
grad_f(M, p) = -A * p + p * (p' * A * p)
Hess_f(M, p, X) = -A * X + p * p' * A * X + X * p' * A * p

p₀ = Matrix{Float64}(I, n, n)[:, 1:k]
M2 = TangentSpace(M, copy(M, p₀))
mho = ManifoldHessianObjective(f, grad_f, Hess_f)
parc = adaptive_regularization_with_cubics(
    M,
    f,
    grad_f,
    Hess_f,
    p₀;
    θ=0.5,
    σ=100.0,
    retraction_method=PolarRetraction(),
    return_state=true,
)
phdm = homogeneous_descent(
    M,
    f,
    grad_f,
    Hess_f,
    p₀;
    θ=0.5,
    σ=100.0,
    retraction_method=PolarRetraction(),
    return_state=true,
)

######################################################################################             
# Fourth with approximate Hessian and random point
# todo, not implemented
# Random.seed!(36)
# p4 = homogeneous_descent(M, f, grad_f; θ=0.5, σ=100.0, retraction_method=PolarRetraction())
######################################################################################             
