using Random
using Distributions
using LinearAlgebra

function snu(data::Matrix{Float64}, dir::Vector{Float64})::Float64
    n, _ = size(data)
    proj = data * dir
    sk = skewness(proj)^2
    #return n*sk/6
    return sk
end

Random.seed!(12345)
N=200
td = Gamma(1,1);
tds = rand(td, N);
n=10
diag = collect(n:-1:1)
diag = diag.^2
D = Diagonal(diag); 
d = MvNormal(zeros(n), D);
dat = rand(d, N);
ndat = hcat(dat', tds);

@testset "projectionpursuit" begin
    res = projection_pursuit(ndat, snu, 3)
    @test length(res.object_vals) == 3
    @test findmax(res.proj[:,1])[2] == 11
end