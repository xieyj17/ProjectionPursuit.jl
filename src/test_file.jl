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


@time res=Projection_Pursuit(ndat,knu;par=false)
@time unit_sphere= GenSphere(n^2, p);
function tf(i)
    sleep(i)
    return i+1
end

res = zeros(10)
@time begin
    @sync for i in 1:10
        Threads.@spawn res[i] = tf(i)
    end
end

@time begin
    for i in 1:10
        tr = tf(i)
        res[i] = tr
    end
end


function task()
    x = rand(0.001:0.001:0.05)  # Generate a variable workload
    sleep(x)  # Sleep simulates the workload
    return x  # Return the workload
end
@time begin
    n = 1000  # Number of tasks
    p = zeros(Threads.nthreads())  # Total workload per thread
    @sync for i in 1:n
        Threads.@spawn p[Threads.threadid()] += task()  # Spawn tasks and sum the workload
    end
end

@time begin
    n = 1000  # Number of tasks
    p = zeros(Threads.nthreads())  # Total workload per thread
    for i in 1:n
        p[i%4+1] += task()  # Spawn tasks and sum the workload
    end
end

