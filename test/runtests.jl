using Pkg
Pkg.add(url="https://github.com/xieyj17/ProjectionPursuit.jl")
Pkg.instantiate()

using ProjectionPursuit
using Test


include("test_gensphere.jl")
include("test_projection_pursuit.jl")

print("all tests passed!")