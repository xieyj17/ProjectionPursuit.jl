using Random
using Distributions
using LinearAlgebra
#using PosDefManifold
#using PDMats




n=10
#diag = rand(Uniform(6,10), n)
#Q, _ = qr(randn(n, n)); 
D = Diagonal(n:-1:1); 
#cov_mat = Q*D*Q';

#cov_mat = Float64.(cov_mat)

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


@btime res=Projection_Pursuit(ndat,knu; par=false,fast=false)
@btime res=Projection_Pursuit(ndat,knu; par=true,fast=false)
@btime res=Projection_Pursuit(ndat,knu; par=false,fast=true)
@btime res=Projection_Pursuit(ndat,knu; par=true,fast=true)

@time unit_sphere= FastGenSphere(n^2, p);

using Plots

scatter(tp[:,1],tp[:,2],markersize=3,alpha=.8,legend=false)
histogram(rds)