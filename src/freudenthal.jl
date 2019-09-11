# Construct the Freudenthal triangulation of the belief simplex


"""
    freudenthal_vertices(n::Int64, m::Int64)
Construct the list of Freudenthal vertices in an `n` dimensional space with grid resolution `m`.
The vertices are represented by a list of `n` dimensional vectors.
"""
function freudenthal_vertices(n::Int64, m::Int64)
    V = Vector{Int}[]
    v = Vector{Int}(undef, n)
    v[1] = m
    freudenthal_vertices!(V, v, 2)
    return V
end

function freudenthal_vertices!(V::Vector{Vector{Int64}}, v::Vector{Int64}, i::Int64)
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

"""
    freudenthal_simplex(x::Vector{Int64})
Returns the list of vertices of the simplex of point `x` in the Freudenthal grid.
"""
function freudenthal_simplex(x::Vector{Int64})
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

"""
    barycentric_coordinates(x::Vector{Int64}, V::Vector{Vector{Int64}})
Given a point `x` and its simplex `V` in the Freudenthal grid, returns the barycentric coordinates
of `x` in the grid. `V` must be in the same order as provided by the output of `freudenthal_simplex`
"""
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

"""
    to_belief(x, m)
Transform a point `x` in the Freudenthal space to a point in the belief space. 
`m` is the resolution of the Freudenthal grid.
"""
to_belief(x, m) = (push!(x[1:end-1] - x[2:end], x[end]))./x[1]

"""
    to_freudenthal(b, m::Int64) 
Transform a point `b` in the belief space to a point in the Freudenthal space.
`m` is the resolution of the Freudenthal grid.
"""
to_freudenthal(b, m::Int64) = [sum(b[k] for k in i : length(b))*m for i in 1 : length(b)]

"""
    freudenthal_simplex_and_coords!(x::AbstractArray{Int64}, V::Vector{Vector{Int64}}, λ::Vector{Float64})
Fills `V` and `λ` with the simplex points in the Freudenthal space and associated coordinates respectively.
"""
function freudenthal_simplex_and_coords!(x::AbstractArray{Float64}, V::Vector{Vector{Int64}}, λ::Vector{Float64})
    n = length(x)
    V[1] = floor.(Int, x)
    d = x - V[1]
    p = sortperm(d, rev=true)
    for i in 2 : n+1
        copyto!(V[i], V[i-1])
        V[i][p[i-1]] += 1
    end
    λ[n+1] = d[p[n]]
    for i in n:-1:2
        λ[i] = d[p[i-1]] - d[p[i]]
    end
    λ[1] = 1.0 - sum(λ[2:end])
    return V, λ 
end

"""
    freudenthal_matrix_inv(n::Int64, m::Int64)
returns the inverse of the matrix used to switch from Freudenthal space to belief space.
Let `IFM = freudenthal_matrix_inv(n, m)`, then `x = IFM * b`
"""
function freudenthal_matrix_inv(n::Int64, m::Int64)
    return m .* sparse(UnitUpperTriangular(ones(n,n)))
end

"""
    to_freudenthal_batch(B::AbstractArray, m::Int64)
Given a batch of belief points `B`, returns the corresponding points in the Freudenthal space.
"""
function to_freudenthal_batch(B::AbstractArray, m::Int64)
    ns = size(B, 1)
    IFM = freudenthal_matrix_inv(ns, m)
    BB = reshape(B, (ns, :))
    X = similar(BB)
    for i=1:size(BB, 2)
        X[:, i] = IFM * view(BB, :, i)
    end
    return reshape(X, size(B))
end