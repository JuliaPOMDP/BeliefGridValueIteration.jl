"""
    BeliefGridValueIterationSolver

An offline POMDP solver from "Computationally Feasible Bounds for Partially Observed Markov Decision Processes" (1991), by W. S. Lovejoy.
It computes an upper bound on the value function by performing value iteration on a discretized belief space.

# Options
- `m::Int64 = 1` Granularity of the belief grid for the triangulation
- `precision::Float64 = 0.0` The solver stops when the desired convergence precision is reached
- `max_iterations::Int64 = 100` Number of iteration of value iteration
- `verbose::Bool = false` whether or not the solver prints information
"""
@with_kw mutable struct BeliefGridValueIterationSolver <: Solver
    m::Int64 = 1
    precision::Float64 = 0.0
    max_iterations::Int64 = 100
    verbose::Bool = false
end


"""
    update_batch(B::Vector{Vector{R}}, pomdp::SparseTabularPOMDP) where R <: Real
Compute the next belief point starting from B, for every action and observation in the POMDP.
Return a 4 dimensional array of size `n_states x n_belief_points x n_actions x n_observations` 
"""
function update_batch(B::Vector{Vector{R}}, pomdp::SparseTabularPOMDP) where R <: Real
    𝒮, 𝒜, 𝒪 = states(pomdp), actions(pomdp), observations(pomdp)
    T, O, γ = pomdp.T, pomdp.O, pomdp.discount
    nS, nA, nO = length(𝒮), length(𝒜), length(𝒪)
    
    Bp = zeros(nS, length(B), nA, nO)
    for a in 𝒜
        for o in 𝒪
            OT = view(O[a], :, o) .* T[a]'
            for (i,b) in enumerate(B)
                update_single!(OT, b, @view Bp[:, i, a, o])
            end
        end
    end
    return Bp
end

"""
    update_single!(OT::AbstractArray, b::Vector{R}, bp::AbstractArray) where R<:Real    
update a single belief point using a precomputed observation * transition matrix for a given o,a pair
"""
function update_single!(OT::AbstractArray, b::Vector{R}, bp::AbstractArray) where R<:Real
    bp[:] = OT * b
    if sum(bp) ≈ 0.0
        fill!(bp, 1)
    end
    return normalize!(bp, 1)
end 

"""
    lovejoy_upper_bound_data(pomdp::SparseTabularPOMDP, m::Int64, verbose=false)
Precompute useful quantities before computing the upper bound:
- Construct the belief space triangulation
- Store mapping from vertices to index 
- Compute the next belief points
- Compute the coordinates of the next belief points in the grid
- Precompute R(b, a)
- Precompute Pr(o | b, a) 
"""
function lovejoy_upper_bound_data(pomdp::SparseTabularPOMDP, m::Int64, verbose=false)
    
    # extract pomdp info
    𝒮, 𝒜, 𝒪 = states(pomdp), actions(pomdp), observations(pomdp)
    T, R, O, γ = pomdp.T, pomdp.R, pomdp.O, pomdp.discount
	nS, nA, nO = length(𝒮), length(𝒜), length(𝒪)
    
    # compute belief grid
    V = freudenthal_vertices(nS, m)
    Vmap = Dict{Vector{Int}, Int}(zip(V, 1:length(V)))
    B = [to_belief(v,m) for v in V]
    
    
    verbose ? @printf("Problem size: |S| = %d | |A| = %d, |O| = %d \n", nS, nA, nO) : nothing
    true ? @printf("Granularity %d resulting in %d belief points \n", m, length(B)) : nothing

    # Precompute R(b, a)
	Rba = Array{Float64}(undef, length(B), nA)
    for (bi,b) in enumerate(B)
        for (ai,a) in enumerate(𝒜)
            Rba[bi, ai] = R[:, a]⋅b
        end
    end
    
    # Next belief points
    verbose ? @printf("Computing next belief points...") : nothing
    Bp = update_batch(B, pomdp)
    Xp = to_freudenthal_batch(Bp, m)
    
    # Precompute vertices and coordinates of the next belief points
    Λ′ = [sparsevec([], Float64[], length(V)) for b=1:length(B) for a=1:nA for o=1:nO]
    Λ′ = reshape(Λ′, length(B), nA, nO)

    Vsub = [Vector(zeros(Int, nS)) for i=1:nS+1]
    λsub = zeros(Float64, nS+1)
    p = verbose ? Progress(length(B), desc="Computing coordinates associated coordinates") : nothing
    for bi in 1:length(B)
        for oi in 1:nO
            for ai in 1:nA

                x′ = view(Xp, :, bi, ai, oi)
                V′, λ′ = freudenthal_simplex_and_coords!(x′, Vsub, λsub)

                for (i,v′) in enumerate(V′)

                    j = get(Vmap, v′, nothing)

                    if j != nothing
                        j = Vmap[v′]
                        Λ′[bi,ai,oi][j] = λ′[i]
                    end

                end

            end
        end
        verbose ? next!(p) : nothing
    end
    
    # Precompute Pr(o | b, a)
    Poba = Array{Float64}(undef, length(B), nA, nO)
    Psao = [O[ai]'*T[ai]' for (ai, a) in enumerate(𝒜)]
    for (bi,b) in enumerate(B)
		for (ai,a) in enumerate(𝒜)
			for (oi,o) in enumerate(𝒪)

                Poba[bi, ai, oi] = dot(b, view(Psao[ai], oi, :))

            end
        end
    end

    return (B, Rba, Poba, Λ′, Vmap)
end


function qvalue!(pomdp, Rba, Poba, L, U, qvals)
    for a in 1:length(actions(pomdp))
        qvals[:, a] = view(Rba, :, a) + sum(discount(pomdp) * view(Poba, :, a, :) .* dot.(view(L, :, a, :), Ref(U)), dims=2)
    end
end

"""
    lovejoy_upper_bound(pomdp, m, ϵ, k_max, verbose=false)
Construct the belief grid and perform some preprocessing operations first (see lovejoy_upper_bound_data).
Then run vectorized value iteration over the belief grid.
Returns the value at each belief point, the associated best action, and a mapping between belief points (in the freudenthal space) and their index.
"""
function lovejoy_upper_bound(pomdp, m, ϵ, k_max, verbose=false)
    𝒮, 𝒜, 𝒪 = states(pomdp), actions(pomdp), observations(pomdp)
    T, R, O, γ = pomdp.T, pomdp.R, pomdp.O, pomdp.discount
    t_tot = 0.0
    verbose ? @printf("Preprocessing discretized belief space... \n") : nothing
    t_preproc = @elapsed begin
        B, Rba, Poba, Λ′, Vmap = lovejoy_upper_bound_data(pomdp, m, verbose)
        Rba = sparse(Rba)
    end
    t_tot += t_preproc
    verbose ? @printf("Finish Preprocessing in %3.3f s \n", t_preproc) : nothing
    
    verbose ? @printf("Starting Dynamic Programming... \n") : nothing

    U = fill(maximum(R), length(B))
    U = zeros(length(B))
    U′ = similar(U)
    qvals = fill(maximum(R), length(B), length(𝒜))
    ΔU_max = Inf
    ϵ_k = Inf
    π = fill(first(𝒜), length(B))
    ΔR_max = maximum(R) - minimum(R)

    for k in 2 : k_max
        t_it = @elapsed begin

        qvalue!(pomdp, Rba, Poba, Λ′, U, qvals)

        U′ = dropdims(maximum(qvals, dims=2), dims=2)

        ΔU_max = norm(U - U′, Inf)
        U = U′
        π = dropdims(getindex.(argmax(round.(qvals, digits=20), dims=2), 2), dims=2)

		ϵ_k = max(γ^k/(1-γ)*ΔR_max,
		          γ^(k+1)/(1-γ)^2*ΔR_max,
                  ΔU_max)

        end #@elapsed

        t_tot += t_it
        verbose ? @printf("iteration %d / %d : precision %3.4f | it time %3.3f | total time %3.3f \n", k-1, k_max, ΔU_max, t_it, t_tot) : nothing
        if ϵ_k ≤ ϵ
            break
        end
    end
    
	return (U, π, Vmap)
end


function POMDPs.solve(solver::BeliefGridValueIterationSolver, pomdp::POMDP)
    spomdp = SparseTabularPOMDP(pomdp)
    U, π, Vmap =  lovejoy_upper_bound(spomdp, solver.m, solver.precision, solver.max_iterations, solver.verbose)
    act_space = ordered_actions(pomdp)
    pol = act_space[π]
    return BeliefGridValueIterationPolicy(solver.m, Vmap, U, pol)
end 

POMDPLinter.@POMDP_require solve(solver::BeliefGridValueIterationSolver, pomdp::POMDP) begin
    @subreq SparseTabularPOMDP(pomdp)
end

"""
    BeliefGridValueIterationPolicy{A}
A Policy object containing a belief grid and the value at each belief points.

# Fields
- `m::Int64` The granularity of the belief grid
- `Vmap::Dict{Vector{Int}, Int}` The mapping from vertices in the grid (freudenthal space) to indices in the value vector
- `val::Vector{Float64}` The values for each belief points
- `pol::Vector{A}` The action at each belief point.
"""
struct BeliefGridValueIterationPolicy{A} <: Policy
    m::Int64
    Vmap::Dict{Vector{Int}, Int}
    val::Vector{Float64}
    pol::Vector{A}
end

function POMDPs.value(p::BeliefGridValueIterationPolicy, b::Vector{Float64})
    x = to_freudenthal(b, p.m)
    n = length(x)
    S = [Vector(zeros(Int, n)) for i=1:n+1]
    C = zeros(n+1)
    freudenthal_simplex_and_coords!(x, S, C)
    val = 0.0
    for (i, s) in enumerate(S)
        j = get(p.Vmap, s, nothing)
        if j != nothing
            val += p.val[j] * C[i]
        end
    end
    return val
end
