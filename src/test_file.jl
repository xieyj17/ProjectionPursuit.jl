using DataFrames
using Random
using Distributions
using LinearAlgebra, PosDefManifold
using PDMats


cov_mat = Matrix(randP(10));



n=10
diag = rand(Uniform(6,10), n)
Q, _ = qr(randn(n, n)); 
D = Diagonal(diag); 
cov_mat = Q*D*Q'

cov_mat = Float32.(cov_mat)

d = MvNormal(zeros(10), cov_mat);

dat = rand(d, 100);

Pingouin.anderson(dat[1,:], "norm")

td = TDist(2);
tds = rand(td, 100);

ndat = hcat(dat', tds);


using MultivariateStats
M = fit(PCA, ndat; maxoutdim=10)
pd = projection(M);

using HypothesisTests

ts = pd[:,1];
nts = (ts .- mean(ts)) / std(ts)

s = ExactOneSampleKSTest(nts, Normal(0,1))
pvalue(s)

ntds = (tds .- mean(tds)) / std(tds)
pvalue(ExactOneSampleKSTest(ntds, Normal(0,1)))