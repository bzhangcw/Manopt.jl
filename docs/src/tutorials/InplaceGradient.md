Speedup using Inplace Evaluation
================

When it comes to time critital operations, a main ingredient in Julia is given by
mutating functions, i.e. those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!.html)
and compute the mean of some points, only that here we use the sphere $\mathbb S^{30}$
and $n=800$ points.

From the aforementioned example.

We first load all necessary packages.

``` julia
using Manopt, Manifolds, Random, BenchmarkTools
Random.seed!(42);
```

And setup our data

``` julia
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
p = zeros(Float64, m + 1)
p[2] = 1.0
data = [exp(M, p, σ * rand(M; vector_at=p)) for i in 1:n];
```

## Classical Definition

The variant from the previous tutorial defines a cost $f(x)$ and its gradient $\operatorname{grad}f(p)$
““”

``` julia
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)))
```

    grad_f (generic function with 1 method)

We further set the stopping criterion to be a little more strict. Then we obtain

``` julia
sc = StopWhenGradientNormLess(1e-10)
p0 = zeros(Float64, m + 1); p0[1] = 1/sqrt(2); p0[2] = 1/sqrt(2)
m1 = gradient_descent(M, f, grad_f, p0; stopping_criterion=sc);
```

We can also benchmark this as

``` julia
@benchmark gradient_descent($M, $f, $grad_f, $p0; stopping_criterion=$sc)
```

    BenchmarkTools.Trial: 78 samples with 1 evaluation.
     Range (min … max):  58.693 ms … 83.403 ms  ┊ GC (min … max):  9.66% … 25.11%
     Time  (median):     63.903 ms              ┊ GC (median):    13.39%
     Time  (mean ± σ):   64.770 ms ±  4.887 ms  ┊ GC (mean ± σ):  12.71% ±  3.19%

         █▆    ▁  ▆▃▆█▃ ▁    ▁    ▁▁                               
      ▇▇▄██▄▇▄▄█▇▇█████▄█▄▄▄▁█▄▁▄▁██▁▁▇▁▁▁▄▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▄▄▁▁▄ ▁
      58.7 ms         Histogram: frequency by time          80 ms <

     Memory estimate: 203.70 MiB, allocs estimate: 745626.

## In-place Computation of the Gradient

We can reduce the memory allocations by implementing the gradient to be evaluated in-place.
We do this by using a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
The motivation is twofold: on one hand, we want to avoid variables from the global scope,
for example the manifold `M` or the `data`, being used within the function.
Considering to do the same for more complicated cost functions might also be worth pursuing.

Here, we store the data (as reference) and one introduce temporary memory in order to avoid
reallocation of memory per `grad_distance` computation. We get

``` julia
struct GradF!{TD,TTMP}
    data::TD
    tmp::TTMP
end
function (grad_f!::GradF!)(M, X, p)
    fill!(X, 0)
    for di in grad_f!.data
        grad_distance!(M, grad_f!.tmp, di, p)
        X .+= grad_f!.tmp
    end
    X ./= length(grad_f!.data)
    return X
end
```

For the actual call to the solver, we first have to generate an instance of `GradF!`
and tell the solver, that the gradient is provided in an [`InplaceEvaluation`](https://manoptjl.org/stable/plans/objective/#Manopt.InplaceEvaluation).
We can further also use [`gradient_descent!`](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent!) to even work inplace of the initial point we pass.

``` julia
grad_f2! = GradF!(data, similar(data[1]))
m2 = deepcopy(p0)
gradient_descent!(
    M, f, grad_f2!, m2; evaluation=InplaceEvaluation(), stopping_criterion=sc
);
```

We can again benchmark this

``` julia
@benchmark gradient_descent!(
    $M, $f, $grad_f2!, m2; evaluation=$(InplaceEvaluation()), stopping_criterion=$sc
) setup = (m2 = deepcopy($p0))
```

    BenchmarkTools.Trial: 159 samples with 1 evaluation.
     Range (min … max):  29.785 ms … 37.723 ms  ┊ GC (min … max): 0.00% … 13.76%
     Time  (median):     31.089 ms              ┊ GC (median):    0.00%
     Time  (mean ± σ):   31.590 ms ±  1.498 ms  ┊ GC (mean ± σ):  0.80% ±  3.10%

        ▄█▂   ▅                                                    
      ▃▄███▅▅▇███▆▆▆▄▆▄▆▇▅▃▄▆▅▃▅█▄▃▅▄▄▄▇▃▄▅▄▁▁▁▁▁▁▁▃▃▃▁▁▁▁▃▁▁▁▄▁▃ ▃
      29.8 ms         Histogram: frequency by time          36 ms <

     Memory estimate: 4.24 MiB, allocs estimate: 6832.

which is faster by about a factor of 2 compared to the first solver-call.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    0.0