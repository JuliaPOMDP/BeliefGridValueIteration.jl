# A non vectorized version of the algorithm
using LinearAlgebra
using POMDPs
using POMDPTools

function freudenthal_vertices!(V, v, i)
    n = length(v)
    if i > n
        push!(V, copy(v))
        return
    end
    for k in 0 : v[i-1]
        v[i] = k
        freudenthal_vertices!(V, v, i+1)
    end
end

function freudenthal_vertices(n, m)
    V = Vector{Int}[]
    v = Vector{Int}(undef, n)
    v[1] = m
    freudenthal_vertices!(V, v, 2)
    return V
end

function freudenthal_simplex(x)
    n = length(x)
    V = Vector{Vector{Int}}(undef, n+1)
    V[1] = floor.(Int, x)
    d = x - V[1]
    p = sortperm(d, rev=true)
    for i in 2 : n+1
        V[i] = copy(V[i-1])
        V[i][p[i-1]] += 1
    end
    return V
end

function barycentric_coordinates(x, V)
    d = x - V[1]
    p = sortperm(d, rev=true)
    n = length(x)
    λ = Vector{Float64}(undef, n+1)
    λ[n+1] = d[p[n]]
    for i in n:-1:2
        λ[i] = d[p[i-1]] - d[p[i]]
    end
    λ[1] = 1 - sum(λ[2:end])
    return λ
end

to_belief(x, m) = (push!(x[1:end-1] - x[2:end], x[end]))./x[1]

to_freudenthal(b, m) = [sum(b[k] for k in i : length(b))*m for i in 1 : length(b)]


function update(b::Vector{R}, pomdp::SparseTabularPOMDP, a, o) where R<:Real
    𝒮, T, O = states(pomdp), pomdp.T, pomdp.O
    b′ = similar(b)
    for (i′, s′) in enumerate(𝒮)
        po = O[a][s′, o]
        b′[i′] = po * sum(T[a][s, s′] * b[i] for (i, s) in enumerate(𝒮))
    end
    if sum(b′) ≈ 0.0
        fill!(b′, 1)
    end
    return normalize(b′, 1)
end

function update2(b::Vector{R}, pomdp::SparseTabularPOMDP, a, o) where R<:Real
    T = transition_matrix(pomdp, a)
    O = view(observation_matrix(pomdp, a), :, o)
    bp = O .* T'* b
    if sum(bp) ≈ 0.0
        fill!(bp, 1)
    end
    return normalize!(bp, 1)
end


function get_lovejoy_upper_bound_data(pomdp, m)
	𝒮, 𝒜, 𝒪 = states(pomdp), actions(pomdp), observations(pomdp)
	T, R, O, γ = pomdp.T, pomdp.R, pomdp.O, pomdp.discount
	nS, nA, nO = length(𝒮), length(𝒜), length(𝒪)
	V = freudenthal_vertices(nS, m)
	B = [to_belief(v,m) for v in V]
	Rba = Array{Float64}(undef, length(B), nA)
    for (bi,b) in enumerate(B)
        for (ai,a) in enumerate(𝒜)
        	Rba[bi,ai] = [R[s, a] for s in 𝒮]⋅b
        end
    end
    B′ = [update(b, pomdp, a, o) for o in 𝒪
							for a in 𝒜
						for b in B]
    B′ = reshape(B′, (length(B),nA,nO))
    @show B′[1,1,1]

    X′ = [to_freudenthal(b′, m) for b′ in B′]
	Λ′ = zeros(Float64, length(B), nA, nO, length(V))
	for bi in 1:length(B)
		for ai in 1:nA
			for oi in 1:nO
				x′ = X′[bi,ai,oi]
				V′ = freudenthal_simplex(x′ .+ eps())
				λ′ = barycentric_coordinates(x′, V′)
				for (i,v′) in enumerate(V′)
					j = findfirst([v == v′ for v in V])
                    if j != nothing
                        Λ′[bi,ai,oi,j] = λ′[i]
                    end
                end
            end
        end
    end
    Poba = Array{Float64}(undef, length(B), nA, nO)
    for (bi,b) in enumerate(B)
		for (ai,a) in enumerate(𝒜)
			for (oi,o) in enumerate(𝒪)
                Poba[bi,ai,oi] = sum(b[si]*sum(O[a][s′,o]*T[a][s,s′]
                                    for (si′,s′) in enumerate(𝒮))
                                       for (si,s) in enumerate(𝒮))
            end
        end
    end
    return (B, Rba, Poba, Λ′)
end

function lovejoy_upper_bound(pomdp, m, ϵ, k_max)
	𝒮, 𝒜, 𝒪 = states(pomdp), actions(pomdp), observations(pomdp)
	T, R, O, γ = pomdp.T, pomdp.R, pomdp.O, pomdp.discount
	B, Rba, Poba, Λ′ = get_lovejoy_upper_bound_data(pomdp, m)

	U = [maximum(Rba[bi,ai] for ai in 1 : length(𝒜))
		 for bi in 1:length(B)]
	U′ = similar(U)
    π = fill(first(𝒜), length(B))
    ΔR_max = maximum(R[s,a] for s in 𝒮 for a in 𝒜)
    		 - minimum(R[s,a] for s in 𝒮 for a in 𝒜)
	for k in 2 : k_max
		for (bi,b) in enumerate(B)
			U′[bi] = -Inf
			for (ai,a) in enumerate(𝒜)
				u′ = Rba[bi,ai] + γ*sum(
                        Poba[bi,ai,oi] * Λ′[bi,ai,oi,:]⋅U
                    for (oi,o) in enumerate(𝒪))
				if u′ > U′[bi]
					U′[bi], π[bi] = u′, a
				end
			end
		end
        ΔU_max = norm(U - U′, Inf)
		U[:] = U′
		ϵ_k = max(γ^k/(1-γ)*ΔR_max,
		          γ^(k+1)/(1-γ)^2*ΔR_max,
		          ΔU_max)
        if ϵ_k ≤ ϵ
            break
        end
	end
	return (U,π)
end
