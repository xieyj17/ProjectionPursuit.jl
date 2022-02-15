module ProjectionPursuit

using Distributed
include("sphere_optimize.jl")

# Write your package code here.

function Projection_Pursuit(data::Matrix{Float64}, object_fun::Function; n_of_candidate::Int64=5 ,unit_sphere=nothing, fnscale::Int64=-1, par::Bool=true)
    n, p = size(data)
    if isnothing(unit_sphere)
        unit_sphere= GenSphere(n^2, p)
    end
    r = map(x->object_fun(data, Vector(x)), eachrow(unit_sphere))
    proj_dirs = hcat(unit_sphere, r)
    proj_dirs = proj_dirs[sortperm(proj_dirs[:, p+1], rev=true), :]
    candidate_dirs = proj_dirs[1:n_of_candidate, 1:p]
    if par
        np = Distributed.nprocs()
        Distributed.addprocs(np)
        opt_objs = pmap(x->Sphere_Optimize(data, Vector(x), object_fun;fnscale), eachrow(candidate_dirs))
    else
        opt_objs = map(x->Sphere_Optimize(data, Vector(x), object_fun;fnscale), eachrow(candidate_dirs))
    end
    max_var = maximum([o.var for o in opt_objs])
    s = randn(p)
    for o in collect(opt_objs)
        if o.var == max_var
            s = o.s
        end
    end
    return Sphere_Optimize_Res(max_var, s)
end


export Projection_Pursuit, Sphere_Optimize, ToSphere, FromSphere, GenSphere,
        Sphere_Optimize_Res

end
