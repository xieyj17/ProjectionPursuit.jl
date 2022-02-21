using Sobol
using QuadGK
using Roots
using Optim

function _y1(x::Vector{Float64})::Matrix{Float64}
    r1 = [cos(2*pi*s) for s in x]
    r2 = [sin(2*pi*s) for s in x]
    return hcat(r1, r2)
end

function _y2(x::Vector{Float64}, y::Matrix{Float64})::Matrix{Float64}
    ya = y[:,1]
    yb = y[:,2]
    p1a = [sqrt(1 - (1 - 2*r)^2) * s for (r,s) in zip(x, ya)]
    p1b = [sqrt(1 - (1 - 2*r)^2) * s for (r,s) in zip(x, yb)]
    p2 = [1 - 2*r for r in x]
    return hcat(p1a, p1b, p2)
end

function _b_inside(a::Float64, b::Float64)
    function tf(u::Float64)
        return u^(a-1) * (1-u)^(b-1)
    end
    return tf
end

function _bz(a::Float64, b::Float64, z::Float64)
    tf = _b_inside(a,b)
    inte, err = quadgk(tf, 0, z, rtol=1e-3)
    return inte
end

function _bd(a::Float64, b::Float64)
    return _bz(a,b,float(1))
end

function _hd(x::Float64, d::Float64)
    ix = _bz(d/2, d/2, x) / _bd(d/2, d/2)
    return ix
end

function _hd_inv(y::Float64, d::Int64)
    function tf(x::Float64)
        return _hd(x, Float64(d)) - y
    end
    res = find_zero(tf, (0, 1))
    return res
end

function yd(x::Vector{Float64}, y::Matrix{Float64}, d::Int64)
    nr, nc = size(y)
    pr = [sqrt(1 - (1 - 2*_hd_inv(r, d))^2) for r in x]
    res = zeros(nr, nc)
    for i in 1:nc
        ty = y[:,i]
        tp = [r * s for (r,s) in zip(pr, ty)]
        res[:,i] = tp
        #append!(res, tp)
    end
    #tres = reshape(res, (nr, nc))
    p2 = [1 - 2*_hd_inv(r, d) for r in x]
    return hcat(res, p2)
end


"""
    gensphere(N::Int64, d::Int64)

Generate low discrepancy sequences of length N on a d dimension unit sphere.

Roughly speaking, a low discrepancy sequence is a sequence of samples of a random variable
    has the property that the number of points within a region is proportional to the size
    of the region.

This function returns a N-by-d matrix, where each row represents a sample of a random 
    vector that is uniformly distributed on a d-dimensional unit sphere.

This function is based on Brauchart, Johann S., Josef Dick, and Lou Fang. 
    "Spatial low-discrepancy sequences, spherical cone discrepancy,
     and applications in financial modeling." 
     Journal of Computational and Applied Mathematics 286 (2015): 28-53.

"""
function gensphere(N::Int64, d::Int64)::Matrix{Float64}
    s = SobolSeq(d-1)
    rds = hcat([next!(s) for i = 1:N] ...)'
    prj1 = _y1(rds[:,1])
    prj2 = _y2(rds[:,2], prj1)
    tp = prj2
    for j in 3:(d-1)
        np = yd(rds[:,j], tp, j)
        tp = np
    end
    return tp
end

"""
    fastgensphere(N::Int64, d::Int64; par::Bool=true)

A faster way to generate lds of length N on a d dimension unit sphere.

Similar to `gensphere()`. This function is much faster and support parallelism, 
but may degenarate when d is large.
"""

function fastgensphere(N::Int64, d::Int64; par::Bool=true)::Matrix{Float64}
    eps=1e-10
    s = SobolSeq(d)
    rds = hcat([next!(s) for i = 1:(N+1)] ...)'
    rds = (rds .*2).-1 
    tp = Matrix{Float64}(undef, (N,d))
    if par
        @sync for j in 2:(N+1)
            Threads.@spawn tp[(j-1),:] = (rds[j,:]) ./ sqrt(sum(rds[j,:].^2).+eps)
        end
    else
        for j in 2:(N+1)
            tp[(j-1),:] = (rds[j,:]) ./ sqrt(sum(rds[j,:].^2).+eps)
        end
    end
    return tp
end

function _regularizedtheta!(theta::Vector{Float64})
    n = length(theta)
    for i in 1:(n-1)
        while theta[i] < 0
            theta[i] += π
        end
        theta[i] = theta[i] % π
    end
    while theta[n] < 0
        theta[n] += 2*π
    end
    theta[n] = theta[n] % π
end

function _medfromsphere(theta::Vector{Float64}, i::Int64)
    res = 1
    if i <= length(theta)
        temp_theta = theta[1:i]
        if i == 1
            res = cos(temp_theta[1])
        else
            for p in 1:(i-1)
                res *= sin(temp_theta[p])
            end
            res *= cos(temp_theta[i])
        end
    else
        temp_theta = theta
        for p in 1:(i-1)
            res *= sin(temp_theta[p])
        end
    end
    return res
end

"""
    fromsphere(theta::Vector{Float64})

Transform a d-dimensional spherical coordinate to the corresponding
d+1 Cartesian coordinate.

See also [`tosphere`](@ref tosphere).
"""

function fromsphere(theta::Vector{Float64})
    s = zeros(length(theta)+1)
    _regularizedtheta!(theta)
    for i in 1:length(s)
        s[i] = _medfromsphere(theta, i)
    end
    return s
end

"""
    tosphere(s::Vector{Float64})

Transform a d-dimensional Cartesian coordinate to the corresponding
d-1 spherical coordinate.

See also [`fromsphere`](@ref fromsphere).
"""

function tosphere(s::Vector{Float64})
    n = length(s)
    theta = zeros(n-1)
    
    for i in 1:(n-1)
        ts = s[i:n]
        theta[i] = acos(s[i] / sqrt(sum(ts.^2)))
    end

    if s[n] < 0
        theta[n-1] = 2*π - theta[n-1]
    end
    
    return theta
end

function _residualsphere(sphere::Matrix{Float64}, dir::Vector{Float64};par=true)
    eps = 1e-10
    proj_val = sphere * dir
    residuals = Matrix{Float64}(undef, size(sphere))
    for i in 1:size(sphere)[1]
        residuals[i,:] = sphere[i,:] - proj_val[i] * dir
    end
    if par
        @sync for j in 1:size(sphere)[1]
            Threads.@spawn residuals[j,:] = (residuals[j,:]) ./ sqrt(sum(residuals[j,:].^2).+eps)
        end
    else
        for j in 1:size(sphere)[1]
            residuals[j,:] = (residuals[j,:]) ./ sqrt(sum(residuals[j,:].^2).+eps)
        end
    end
    return residuals
end

function _orthogonal_dir(dir::Vector{Float64}, proj_dirs::Matrix{Float64})
    proj_val = dir' * proj_dirs
    for j in 1:length(proj_val)
        dir = dir - proj_dirs[:,j]*proj_val[j]
    end
    return dir
end


struct SphereOptimizeRes
    var::Float64
    s::Vector{Float64}
end

function sphere_optimize_emb(data::Matrix{Float64}, s::Vector{Float64}, object_fun::Function;fnscale::Int64=-1)
    theta = tosphere(s)
    _regularizedtheta!(theta)
    temp_fn(theta) = object_fun(data, fromsphere(theta))*fnscale
    k = optimize(temp_fn, theta, LBFGS())
    ntheta = Optim.minimizer(k)
    _regularizedtheta!(ntheta) 
    return SphereOptimizeRes(object_fun(data, fromsphere(ntheta)), fromsphere(ntheta))
end

function sphere_optimize_emb(data::Matrix{Float64}, s::Vector{Float64}, object_fun::Function, proj_dirs::Matrix{Float64};
    fnscale::Int64=-1)
    theta = tosphere(s)
    _regularizedtheta!(theta)
    temp_fn(theta) = object_fun(data, _orthogonal_dir(fromsphere(theta), proj_dirs))*fnscale
    k = optimize(temp_fn, theta, LBFGS())
    ntheta = Optim.minimizer(k)
    _regularizedtheta!(ntheta) 
    return SphereOptimizeRes(object_fun(data, fromsphere(ntheta)), fromsphere(ntheta))
end

