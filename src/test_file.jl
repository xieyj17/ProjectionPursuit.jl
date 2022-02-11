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


td = TDist(2);
tds = rand(td, 100);

ndat = hcat(dat', tds);


using MultivariateStats
M = fit(PCA, ndat; maxoutdim=10)
pd = projection(M);

using HypothesisTests

s = JarqueBeraTest(pd[:,1])
pvalue(s)
s.JB

pvalue(JarqueBeraTest(pd[:,10]))
pvalue(JarqueBeraTest(tds))

for i in 1:10
    println(pvalue(JarqueBeraTest(pd[:,i])))
end

