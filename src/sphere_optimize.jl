#using Distributions
using Sobol
using QuadGK
using Roots
#using Distributed
using Optim

function y1(x::Vector{Float64})::Matrix{Float64}
    r1 = [cos(2*pi*s) for s in x]
    r2 = [sin(2*pi*s) for s in x]
    return hcat(r1, r2)
end

function y2(x::Vector{Float64}, y::Matrix{Float64})::Matrix{Float64}
    ya = y[:,1]
    yb = y[:,2]
    p1a = [sqrt(1 - (1 - 2*r)^2) * s for (r,s) in zip(x, ya)]
    p1b = [sqrt(1 - (1 - 2*r)^2) * s for (r,s) in zip(x, yb)]
    p2 = [1 - 2*r for r in x]
    return hcat(p1a, p1b, p2)
end

function b_inside(a::Float64, b::Float64)
    function tf(u::Float64)
        return u^(a-1) * (1-u)^(b-1)
    end
    return tf
end

function bz(a::Float64, b::Float64, z::Float64)
    tf = b_inside(a,b)
    inte, err = quadgk(tf, 0, z, rtol=1e-3)
    return inte
end

function bd(a::Float64, b::Float64)
    return bz(a,b,float(1))
end

function hd(x::Float64, d::Float64)
    ix = bz(d/2, d/2, x) / bd(d/2, d/2)
    return ix
end

function hd_inv(y::Float64, d::Int64)
    function tf(x::Float64)
        return hd(x, Float64(d)) - y
    end
    res = find_zero(tf, (0, 1))
    return res
end

function yd(x::Vector{Float64}, y::Matrix{Float64}, d::Int64)
    nr, nc = size(y)
    pr = [sqrt(1 - (1 - 2*hd_inv(r, d))^2) for r in x]
    res = zeros(nr, nc)
    for i in 1:nc
        ty = y[:,i]
        tp = [r * s for (r,s) in zip(pr, ty)]
        res[:,i] = tp
        #append!(res, tp)
    end
    #tres = reshape(res, (nr, nc))
    p2 = [1 - 2*hd_inv(r, d) for r in x]
    return hcat(res, p2)
end



function gen_Sphere(N::Int, d::Int)::Matrix
    s = SobolSeq(d-1)
    rds::Matrix{Float64} = hcat([next!(s) for i = 1:N] ...)'
    prj1::Matrix{Float64} = y1(rds[:,1])
    prj2::Matrix{Float64} = y2(rds[:,2], prj1)
    tp::Matrix{Float64} = prj2
    for j in 3:(d-1)
        np::Matrix = yd(rds[:,j], tp, j)
        tp = np
    end
    return tp
end


function regularized_Theta(ang::Float64)
    ang = ang % (2*pi)
    while ang < 0
        ang += 2*pi
    end
    ang = ang % (2*pi)
    return ang
end

function med_from_Sphere(theta::Vector{Float64}, i::Int64)
    res = 1
    if i <= length(theta)
        temp_theta = theta[1:i]
        if i == 1
            res = cos(temp_theta[1])
        else
            for p in 1:(i-1)
                res = res * sin(temp_theta[p])
            end
            res = res * cos(temp_theta[i])
        end
    else
        temp_theta = theta
        for p in 1:(i-2)
            res *= sin(temp_theta[p])
        end
        res = res * cos(temp_theta[i-1])
    end
    return res
end

function from_Sphere(theta::Vector{Float64})
    s = zeros(length(theta)+1)
    for k in 1:length(theta)
        theta[k] = regularized_Theta(theta[k])
    end
    for i in 1:length(s)
        s[i] = med_from_Sphere(theta, i)
    end
    return s
end


function to_Sphere(s::Vector{Float64})
    theta = zeros(length(s)-1)
    theta[1] = acos(s[1])
    for i in 2:length(theta)
        theta[i] = acos(s[i] / sqrt(sum([w^2 for w in s[i:length(s)]])))
    end
    if s[length(s)] < 0
        theta[length(theta)] = 2*pi - theta[length(theta)]
    end
    if ismissing(theta)
        theta[[ismissing(x) for x in theta]] = 0
    end
    return theta
end

function Sphere_Optimize(par, fn, neighbor = NaN)
    function temp_fn(t)
        s = from_Sphere(t)
        res = fn(s)
        return res
    end

    theta = to_Sphere(par)
    k = optimize(temp_fn, theta, LBFGS())
    ntheta = Optim.minimizer(k)
    ntheta = [regularized_Theta(t) for t in ntheta]
    par = to_Sphere(ntheta)
    nv = Optim.minimum(k)

    return nv
end