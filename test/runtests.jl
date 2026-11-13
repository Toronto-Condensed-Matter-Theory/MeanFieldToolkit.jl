using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, JLD2, Logging, Random
using Test

##### the solvers log every iteration; only show warnings and errors
global_logger(ConsoleLogger(stderr, Logging.Warn))

include("models.jl")
##### model builders shared with the example scripts in Sample/
include(joinpath(@__DIR__, "..", "Sample", "common.jl"))

@testset "MeanFieldToolkit.jl" begin
    include("test_decompose.jl")
    include("test_bonds.jl")
    include("test_energies.jl")
    include("test_hubbard.jl")
    include("test_io.jl")
    include("test_bdg.jl")
    include("test_examples.jl")
end
