using DataFrames
using Random
using Distributions
using LinearAlgebra, PosDefManifold
using PDMats


cov_mat = Matrix(randP(10));



n=10
diag = rand(Uniform(6,10), n)
Q, _ = qr(randn(n, n)); 
D = Diagonal(n:-1:1); 
cov_mat = Q*D*Q';

cov_mat = Float64.(cov_mat)

d = MvNormal(zeros(n), D);

dat = rand(d, 100);


td = TDist(5);
tds = rand(td, 100);

ndat = hcat(dat', tds);


using MultivariateStats
M = fit(PCA, ndat; pratio=0.9)
pd = projection(M);

using HypothesisTests

s = JarqueBeraTest(pd[:,1])
pvalue(s)
s.JB

pvalue(JarqueBeraTest(pd[:,10]))
pvalue(JarqueBeraTest(tds))

for i in 1:28
    println(pvalue(JarqueBeraTest(pd[:,i])))
end


lower = [t - 0.2*π for t in theta]
lower = [maximum([t, 0]) for t in lower]
upper = [t + 0.2*π for t in theta]
upper = [minimum([t, 2*π]) for t in upper]

inner_optimizer = GradientDescent()
results = optimize(temp_fn, lower, upper, initial_x, Fminbox(inner_optimizer))


res=Projection_Pursuit(ndat,knu)