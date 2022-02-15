using  MultivariateStats
using HypothesisTests
using Distributions

struct GHHK_res
    test_statistic::Float64
    critical_value::Float64
    pvalue::Float64
end

function GHHK(dat::Matrix{Float64}, α::Float64=0.05, pratio::Float64=0.9)
    M = fit(PCA, dat'; pratio=pratio)
    pca_proj = dat * M.proj
    n, p = size(pca_proj)
    s = 0
    for i in 1:p
        temp_dat = pca_proj[:,i]
        tau = skewness(temp_dat)
        kappa = kurtosis(temp_dat)
        s += (tau^2 /6 + kappa^2 / 24) * n
    end
    temp_chi = Chisq(2*p)
    critical_value = quantile(temp_chi, 1-α)
    pvalue = 1-cdf(temp_chi, s)
    return GHHK_res(s, critical_value, pvalue)
end


struct MAX_res
    test_statistic::Float64
    critical_value::Float64
    pvalue::Float64
end

function MAX(dat::Matrix{Float64}, α::Float64=0.95, pratio::Float64=0.9)
    M = fit(PCA, dat'; pratio=pratio)
    pca_proj = dat * M.proj
    n, p = size(pca_proj)
    s = zeros(p)
    for i in 1:p
        s[i] = JarqueBeraTest(pca_proj[:,i]).JB
    end
    test_stat = maximum(s)
    temp_chi = Chisq(2)
    critical_value = quantile(temp_chi, 1-α^p)
    pvalue = 1-(cdf(temp_chi, test_stat))^p
    return MAX_res(test_stat, critical_value, pvalue)
end

function snu(dat::Matrix{Float64}, dir::Vector{Float64})::Float64
    n, _ = size(dat)
    proj = dat * dir
    sk = skewness(proj)^2
    return n*sk/6
end

function knu(dat::Matrix{Float64}, dir::Vector{Float64})::Float64
    n, _ = size(dat)
    proj = dat * dir
    ku = abs(kurtosis(proj))
    return sqrt(n/24)*ku
end

function sim_dummy_dat(dat::Matrix{Float64}, B=1000)
    n, p = size(dat)
    sks = zeros(B)
    kus = zeros(B)
    dummy_dist = MvNormal(zeros(p), Diagonal(ones(p)))
    for i in 1:B
        td = rand(dummy_dist, n)
    end
end
