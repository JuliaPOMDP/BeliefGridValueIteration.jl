using Test
using BeliefGridValueIteration
using POMDPModels
using POMDPs
using SARSOP

@testset "basic" begin
    pomdp = TigerPOMDP()
    solver = BeliefGridValueIterationSolver(m = 1, verbose=true)
    # test verbose
    solve(solver, pomdp)
    
    solver.verbose = false
    
    p1 = solve(solver, pomdp)
    
    solver.m = 5
    p5 = solve(solver, pomdp)
    
    b0 = [0.5, 0.5]
    @test value(p1, b0) > value(p5, b0) # finer grid = closer upper bound
end

@testset "requirements" begin 
    @requirements_info BeliefGridValueIterationSolver() TigerPOMDP()
end

@testset "SARSOP comparison" begin 
    pomdp = TigerPOMDP()
    solver = BeliefGridValueIterationSolver(m = 10)

    p1 = solve(solver, pomdp)

    p2 = solve(SARSOPSolver(precision=1e-3, verbose=false), pomdp)

    b0 = [0.5, 0.5]

    value(p1, b0)

    value(p2, b0)
end
