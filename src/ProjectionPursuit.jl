module ProjectionPursuit

include("sphere_optimize.jl")

"""
    ProjectionPursuitRes

Returned object of `projection_pursuit` or `sphere_optimize`.

# Fields
- `object_vals::Vector{Float64}`: projetion pursuit index. Analogous to eigenvalues in PCA.
- `proj::Matrix{Float64}`: projetion pursuit direction. Analogous to eigenvectors in PCA.
"""
struct ProjectionPursuitRes
    object_vals::Vector{Float64}
    proj::Matrix{Float64}
end

"""
    sphere_optimize

Optimize a given objective fucntion constrained on a unit sphere.

See http://hdl.handle.net/10012/16710 for more detail.

# Arguments
- `data::Matrix{Float64}`: each row contains an observation.
- `object_fun::Function`: an arbitrary objective function defined on unit sphere.
- `n_of_candidate::Int64=5`: number of candidate directions in the fine search step.
- `unit_sphere=nothing`: the unit spehre can be pre-generated. By default a unit sphere
    will be generated automatically.
- `proj=nothing`: previously found projection directions. 
    See [`projection_pursuit`](@ref projection_pursuit).
- `fnscale::Int64=-1`: maximize or minimize the objective function. By default 
    equals -1, which means maximize objective function. If the objective function
    needs to be minimized, set to `fnscale=1`.
- `par::Bool=true`: run the program in parallel. 
- `fast::Bool=true`: generate the unit sphere using fast algorithm.
    See [`gensphere`](@ref gensphere) and [`fastgensphere`](@ref fastgensphere).
"""
function sphere_optimize(data::Matrix{Float64}, object_fun::Function; 
    n_of_candidate::Int64=5 ,unit_sphere=nothing, proj=nothing, fnscale::Int64=-1, par::Bool=true, fast::Bool=true)
    n, p = size(data)
    if isnothing(unit_sphere)
        if fast
            unit_sphere= fastgensphere(n^2, p;par)
        else
            unit_sphere= gensphere(n^2, p)
        end
    end

    if par
        nu = size(unit_sphere)[1]
        r = zeros(nu)
        @sync for i in 1:nu
            Threads.@spawn r[i] = object_fun(data, unit_sphere[i,:])
        end
        r = map(x-> isnan(x) ? 0 : x, r)
        r = Vector{Float64}(r)
    else
        r = map(x->object_fun(data, Vector(x)), eachrow(unit_sphere))
    end
    proj_dirs = hcat(unit_sphere, r)
    proj_dirs = proj_dirs[sortperm(proj_dirs[:, p+1], rev=true), :]
    candidate_dirs = proj_dirs[1:n_of_candidate, 1:p]
    
    if isnothing(proj)
        if par 
            opt_objs = Vector{Any}(undef, n_of_candidate)
            @sync for i in 1:n_of_candidate
                Threads.@spawn opt_objs[i] = sphere_optimize_emb(data, Vector(candidate_dirs[i,:]), object_fun;fnscale)
            end
            #opt_objs = pmap(x->Sphere_Optimize(pdata, Vector(x), pobject_fun;fnscale), eachrow(candidate_dirs))        
        else
            opt_objs = map(x->sphere_optimize_emb(data, Vector(x), object_fun;fnscale), eachrow(candidate_dirs))
        end
    else
        if par 
            opt_objs = Vector{Any}(undef, n_of_candidate)
            @sync for i in 1:n_of_candidate
                Threads.@spawn opt_objs[i] = sphere_optimize_emb(data, Vector(candidate_dirs[i,:]), object_fun, proj;fnscale)
            end
            #opt_objs = pmap(x->Sphere_Optimize(pdata, Vector(x), pobject_fun;fnscale), eachrow(candidate_dirs))        
        else
            opt_objs = map(x->sphere_optimize_emb(data, Vector(x), object_fun, proj;fnscale), eachrow(candidate_dirs))
        end
    end    
    
    max_var = maximum([o.var for o in opt_objs])
    s = zeros(p)
    for o in collect(opt_objs)
        if o.var == max_var
            s = o.s
        end
    end
    return SphereOptimizeRes(max_var, s)
end


"""
    projection_pursuit

Conduct dimension reduction corresponding to specified objective function.

See http://hdl.handle.net/10012/16710 for more detail.

# Arguments
- `data::Matrix{Float64}`: each row contains an observation.
- `object_fun::Function`: an arbitrary objective function defined on unit sphere.
- `outdim::Int64`: target dimension of the output.
- `n_of_candidate::Int64=5`: number of candidate directions in the fine search step.
- `unit_sphere=nothing`: the unit spehre can be pre-generated. By default a unit sphere
    will be generated automatically.
- `fnscale::Int64=-1`: maximize or minimize the objective function. By default 
    equals -1, which means maximize objective function. If the objective function
    needs to be minimized, set to `fnscale=1`.
- `par::Bool=true`: run the program in parallel. 
- `fast::Bool=true`: generate the unit sphere using fast algorithm.
    See [`gensphere`](@ref gensphere) and [`fastgensphere`](@ref fastgensphere).
"""
function projection_pursuit(data::Matrix{Float64}, object_fun::Function, outdim::Int64=1; 
    n_of_candidate::Int64=5 ,unit_sphere=nothing, fnscale::Int64=-1, par::Bool=true, fast::Bool=true)

    n, p = size(data)
    if p < outdim
        throw(ArgumentError("Output dimension is greater than data dimension."))
    end

    if isnothing(unit_sphere)
        if fast
            unit_sphere= fastgensphere(n^2, p;par)
        else
            unit_sphere= gensphere(n^2, p)
        end
    end

    object_vals = Vector{Float64}(undef, outdim)
    proj = Matrix{Float64}(undef, (p, outdim))

    t_opt_res = sphere_optimize(data, object_fun;n_of_candidate, unit_sphere, fnscale, par, fast)
    object_vals[1] = t_opt_res.var
    ts = t_opt_res.s
    proj[:,1] = ts
    
    if outdim > 1
        for i in 2:outdim
            unit_sphere = _residualsphere(unit_sphere, ts; par)
            t_opt_res = sphere_optimize(data, object_fun;
                n_of_candidate=n_of_candidate, unit_sphere=unit_sphere, proj=proj[:,1:(i-1)],fnscale=fnscale,
                par=par, fast=fast)
            object_vals[i] = t_opt_res.var
            ts = t_opt_res.s
            proj[:,i] = ts
        end
    end
    return ProjectionPursuitRes(object_vals, proj)
end

export projection_pursuit, sphere_optimize, tosphere, fromsphere, gensphere, fastgensphere,
       SphereOptimizeRes, ProjectionPursuitRes

end
