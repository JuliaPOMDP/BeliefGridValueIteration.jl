# BeliefGridValueIteration

[![CI](https://github.com/JuliaPOMDP/BeliefGridValueIteration.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaPOMDP/BeliefGridValueIteration.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/JuliaPOMDP/BeliefGridValueIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaPOMDP/BeliefGridValueIteration.jl?branch=master)

An offline POMDP solver from "Computationally Feasible Bounds for Partially Observed Markov Decision Processes" (1991), by W. S. Lovejoy.
It computes an upper bound on the value function by performing value iteration on a discretized belief space.

## Installation

Install using the standard package manager:

```julia
using Pkg
Pkg.add("BeliefGridValueIteration")
```

## Usage

```julia
using POMDPs
using POMDPModels # for the tiger pomdp problem
using BeliefGridValueIteration

pomdp = TigerPOMDP()

solver = BeliefGridValueIterationSolver(m = 2, verbose=true)

policy = solve(solver, pomdp)

# Evaluate the value at a given belief point
b0 = [0.5, 0.5]
value(policy, b0)
```

## Documentation

**Solver Options:**

- `m::Int64 = 1` Granularity of the belief grid for the triangulation
- `precision::Float64 = 0.0` The solver stops when the desired convergence precision is reached
- `max_iterations::Int64 = 100` Number of iteration of value iteration
- `verbose::Bool = false` whether or not the solver prints information

**Requirements:**

This should return a list of the following functions to be implemented for your POMDP to be solved by this solver:
```julia
@requirements_info BeliefGridValueIterationSolver() YourPOMDP()
```

## Acknowledgements

The authors thank Tim Wheeler and Mykel Kochenderfer for providing a starter implementation of this code.
