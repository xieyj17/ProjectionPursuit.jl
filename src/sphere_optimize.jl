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



function GenSphere(N::Int, d::Int)::Matrix
    s = SobolSeq(d-1)
    rds = hcat([next!(s) for i = 1:N] ...)'
    prj1 = y1(rds[:,1])
    prj2 = y2(rds[:,2], prj1)
    tp = prj2
    for j in 3:(d-1)
        np = yd(rds[:,j], tp, j)
        tp = np
    end
    return tp
end


function RegularizedTheta!(theta::Vector{Float64})
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

function MedFromSphere(theta::Vector{Float64}, i::Int64)
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

function FromSphere(theta::Vector{Float64})
    s = zeros(length(theta)+1)
    RegularizedTheta!(theta)
    for i in 1:length(s)
        s[i] = MedFromSphere(theta, i)
    end
    return s
end


function ToSphere(s::Vector{Float64})
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


struct Sphere_Optimize_Res
    var::Float64
    s::Vector{Float64}
end

function Sphere_Optimize(data::Matrix{Float64}, s::Vector{Float64}, object_fun::Function;fnscale::Int64=-1)
    theta = ToSphere(s)
    RegularizedTheta!(theta)
    temp_fn(theta) = object_fun(data, FromSphere(theta))*fnscale
    k = optimize(temp_fn, theta, LBFGS())
    ntheta = Optim.minimizer(k)
    RegularizedTheta!(ntheta) 
    return Sphere_Optimize_Res(object_fun(data, FromSphere(ntheta)), FromSphere(ntheta))
end

