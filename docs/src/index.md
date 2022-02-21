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
ProjectionPursuitRes(object_vals, proj)
sphere_optimize(data, object_fun)
projection_pursuit(data, object_fun, outdim)
```

## Index

```@index
```
