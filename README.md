ProjectionPursuit.jl
========

Dimension reduction using projection pursuit in Julia.

<p align="center">
<img src="https://github.com/xieyj17/ProjectionPursuit.jl/raw/main/docs/src/assets/logo.png" width="200"/>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xieyj17.github.io/ProjectionPursuit.jl/)
[![Build Status](https://app.travis-ci.com/xieyj17/ProjectionPursuit.jl.svg?branch=main)](https://app.travis-ci.com/xieyj17/ProjectionPursuit.jl)
[![Coverage](https://codecov.io/gh/xieyj17/ProjectionPursuit.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xieyj17/ProjectionPursuit.jl)
[![Reference](https://img.shields.io/badge/Reference-http%3A%2F%2Fhdl.handle.net%2F10012%2F16710-brightgreen)](http://hdl.handle.net/10012/16710)

(I know the math notation below looks crappy... It's the year of 2022 and GitHub still doesn't support LaTeX...smh)

# Tl;dr

## What does projection pursuit do? 
Simply put, it is complementary to the good old PCA. It allows you to find a lower-dimensional representation of the original data according to your specified objective function (compared with PCA which is restricted to the sample variance). I intended to say replace or substitute PCA but I also want to be humble here because there is a good chance that I might be horribly wrong. 

## Why using Julia? 
It’s lightning-fast. Enough said. 

## What is the next step? 
First thing is to continue improving the computational efficiency. It is always challenging to search in a high-dimensional space. The current version supports multi-threading, but I am still learning how to handle parallelization in Julia properly. Potential GPU support is also on my to-do list. Another thing is to develop several peripheral packages built upon ProjectionPursuit.jl to demonstrate how it can be applied in practice. 


# What is ***Projection Pursuit***?
This terminology might be new to you, but it has been discussed among statisticians since the 70s, including the *great* John Tukey.

To understand what is *projection pursuit*, we should first revist the definition of principal component analysis (PCA).

For a given d-dimensional unit sphere <img src="https://render.githubusercontent.com/render/math?math=u \in U^{d}">, let <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> be the sample variance of the projection scores <img src="https://render.githubusercontent.com/render/math?math=\langle x_1, u \rangle, \ldots, \langle x_n, u \rangle">, where <img src="https://render.githubusercontent.com/render/math?math=\langle \cdot \rangle"> is the inner product. Then PCA tries to find the projection directions that maximize the variance of these projection scores:
![pca](/docs/src/assets/pca.png)

And the definition for *projection pursuit* is very straight forward. Instead of defining <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> to be the sample variance of those scalar projection scores, we allow it to be an arbitrary function that is aligned with the exact problem we want to solve. Hence, PCA is just a special case of *projection pursuit*, when we are only interested in the variance of the data.

# Why you need projection pursuit?
You may ask "why bother allowing an arbitrary definition for the objective function <img src="https://render.githubusercontent.com/render/math?math=Q(u)">?". People seem to be happy using PCA for most of the dimension reduction tasks in the past century. 

But I believe a better question should be "why <img src="https://render.githubusercontent.com/render/math?math=Q(u)"> has to be a measure of variance?". When doing dimension reduction, the projection direction should be directly related to the problem you are working with, rather than blindly using variance for all cases. It's like if you ask me what is 2\* 3, no matter how closely is addition associated with multiplication, no matter how 5 is close to 6, and not matter how faster is '+' than '\*', the correct answer from me should always be 2\*3=6 instead of 2+3=5.

Let me try to convince you why PCA should be concerning with my scratchy logo as another exmaple. You can find random points with three different colors: green, purple, and red. Assume we are some poor creatures who can only process information of one dimension, then the way we understand these colored dots solely depends on how we conduct the dimension reduction for the 2-D data. If the goal is to understand how many clusters are there, but we blindly apply PCA, for which the projection direction is indicated By the blue line on the right. Then the three colored lines on the right show how three clusters would look like after doing PCA. Obviously it's a disaster. They mixed together and there is no way you can understand the clustering structure.

Projection pursuit to the rescue! Clearly the optimal way is to project the data onto the black line in the bottom. In this case we can set the objective function to be the distance between clusters. And you can easily tell that the three clusters are well seperated.

I believe for statisticians, or data scientists or anyone who need to do a dimension reduction, most of the day-to-day problems they are facing do not only aligned with variance. There could be a clear objective function, whether it's prediction error or the level of data drifting or skewness and kurtosis, all these objective functions should be incorporated into the dimension reduction process. The only reason you should to use PCA is when you can explicitly state that your problem is only about the variance of the data. 

# How to use this package?
It's simple. Just use it as PCA, but feed it with an objective function. Here is a toy exmaple.

To install the package:

```julia
using Pkg
Pkg.add("ProjectionPursuit")
using ProjectionPursuit
```

Let's generate the data first with the secret data generating process:

```julia
using Random
using Distributions
using LinearAlgebra
Random.seed!(12345);
N=1500
gamma_d = Gamma(2,1);
gamma_data = rand(gamma_d, N);
n=35;
diag = collect(n+1:-1:2);
diag = diag.^2;
D = Matrix(Diagonal(diag)); 
gaussian_d = MvNormal(zeros(n), D);
gaussian_data = rand(gaussian_d, N);
data = hcat(gaussian_data', gamma_data);
```
Let me explain what happened above. The 36-dimensional random vector is composed of 35 independent Gaussian random variables with standard deviation from 36 to 2, and a Gamma(2,1) random variable with standard deviation of 2. We generate 500 samples of such random vectors.

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

Let's project the data onto the first direction, and compare the density function with that of the Gamma component to see how well projection pursuit can reveal the relevant component of the data:

![gamma_pp](/docs/src/assets/gamma_pp_35.png)

And just in case you are curious what if we use PCA, here is the comparison of density functions. PCA just tells you what is the direction with largest variance, and it totally misses the skewed part. Not surprising at all!

![gamma_pca](/docs/src/assets/gamma_pca_35.png)
