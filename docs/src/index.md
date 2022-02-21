```@meta
CurrentModule = ProjectionPursuit
```

# ProjectionPursuit

Documentation for [ProjectionPursuit](https://github.com/xieyj17/ProjectionPursuit.jl).

```@contents
```

## Functions

```@docs
gensphere(N::Int64, d::Int64)
fastgensphere(N::Int64, d::Int64; par::Bool=true)
fromsphere(theta::Vector{Float64})
tosphere(s::Vector{Float64})
ProjectionPursuitRes
projection_pursuit(data::Matrix{Float64}, sphere_optimize(data::Matrix{Float64}, object_fun::Function; 
    n_of_candidate::Int64=5 ,unit_sphere=nothing, proj=nothing, 
    fnscale::Int64=-1, par::Bool=true, fast::Bool=true)
projection_pursuit(data::Matrix{Float64},               object_fun::Function, 
    outdim::Int64=1; 
    n_of_candidate::Int64=5 ,unit_sphere=nothing, fnscale::Int64=-1, 
    par::Bool=true, fast::Bool=true)
```

## Index

```@index
```
