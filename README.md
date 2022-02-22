ProjectionPursuit.jl
========

Dimension reduction corresponds to specified objective function in Julia.
<p align="center">
<img src="https://github.com/xieyj17/ProjectionPursuit.jl/raw/main/docs/src/assets/logo.png" width="200"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xieyj17.github.io/ProjectionPursuit.jl/)
[![Build Status](https://app.travis-ci.com/xieyj17/ProjectionPursuit.jl.svg?branch=main)](https://app.travis-ci.com/xieyj17/ProjectionPursuit.jl)
[![Coverage](https://codecov.io/gh/xieyj17/ProjectionPursuit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xieyj17/ProjectionPursuit.jl)
[![Reference](https://img.shields.io/badge/Reference-http%3A%2F%2Fhdl.handle.net%2F10012%2F16710-brightgreen)](http://hdl.handle.net/10012/16710)


# What is ***Projection Pursuit***?
This terminology might be new to you, but it has been by statisticians since the 70s, including the great John Tukey.

To understand what is *projection pursuit*, we should first revist the definition of principal component analysis (PCA).

For a given <img src="https://render.githubusercontent.com/render/math?math=u \in U^{d}">, let <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> be the sample variance of <img src="https://render.githubusercontent.com/render/math?math=\langle x_1, u \rangle, \ldots, \langle x_n, u \rangle">, where <img src="https://render.githubusercontent.com/render/math?math=\langle \cdot \rangle"> is the inner product. Then PCA intends to find the following projection directions
![gamma_pca](/docs/src/assets/pca.png)

And the definition for *projection pursuit* is very simple. Instead of defining <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> to be the sample variance of scalar projections, we allow it to be an arbitrary function that is aligned with the exact problem we want to solve. Hence, PCA is just a special case of *projection pursuit*.

# Why you need projection pursuit?
You may ask "why bother allowing an arbitrary definition for the objective function <img src="https://render.githubusercontent.com/render/math?math=Q(u)">?". People seem to be happy using PCA for most of the dimension reduction tasks in the past century. 

But I believe a better question should be "why <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> has to be a measure of variance". Let me try to convince you why PCA should be concerning with my scratchy logo as an exmaple. You can find random points with three different colors: green, purple, and red. Assume we are some poor creatures who can only digest one dimensional information, then the way we understand the data solely based on how we conduct the dimension reduction. If the goal is to understand how many clusters are there, and we blindly apply PCA. The projection direction is indicated in the blue line on the right, and the three colored lines show how three clusters would look like. Obviously it's a disaster. They mixed together and there is no way you can understand the clustering structure.

Projection pursuit to the rescue. Clearly the optimal way is to project the data onto the black line in the bottom. In this case we can set the objective function to be the distance between clusters. And you can easily tell that the three clusters are well seperated.

I believe for statisticians, or data scientists or anyone who need to do a dimension reduction, the day-to-day problem they are facing does not directly related to variance. There should be a clear objective function, where it's prediction error or level of data drifting or skewness and kurtosis, all these objective functions should be incorporated into the dimension reduction. The only reason you should to use PCA is when you can explicitly state that your problem is only about the variance of the data. 

# How to use this package?
It's simple. Just use it as PCA, but feed it with an objective function. Here is a toy exmaple.

Let's generate the data first with the secret data generating process:

```julia
using Random
using Distributions
using LinearAlgebra
Random.seed!(12345);
N=500
gamma_d = Gamma(2,1);
gamma_data = rand(gamma_d, N);
n=10;
diag = collect(n:-1:1);
diag = diag.^2;
D = Matrix(Diagonal(diag)); 
gaussian_d = MvNormal(zeros(n), D);
gaussian_data = rand(d, N);
data = hcat(gaussian_data', gamma_data);
```
Let me explain what happened above. The 11-dimensional random vector is constituted with 10 independent Gaussian random variables with standard deviation from 10 to 1, and a Gamma(1,1) random variable with standard deviation of 1. We generate 200 samples of such random vectors.

Now suppose our goal is to understand whether the data is skewed or not. Clearly the reasonable is a measure of skewness. Here we use the square of skewness as our objective function.
```julia
function snu(data::Matrix{Float64}, dir::Vector{Float64})::Float64
    n, _ = size(data)
    proj = data * dir
    sk = skewness(proj)^2
    return sk
end
```

Then we just feed the data and objective function to `projection_pursuit`, and tell it that we need the first three dimension.
```julia
using ProjectionPursuit
res = projection_pursuit(data, snu, 3)
```

Let's project the data onto the first direction, and compare the density function with the Gamma(1,1) distribution.
![gamma_pp](/docs/src/assets/gamma_pp.png)

And just in case you are curious what if we use PCA compare the density of the first principle with Gamma(1,1) distribution.
![gamma_pca](/docs/src/assets/gamma_pca.png)
