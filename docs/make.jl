using ProjectionPursuit
using Documenter

DocMeta.setdocmeta!(ProjectionPursuit, :DocTestSetup, :(using ProjectionPursuit); recursive=true)

using Documenter
using Demo
makedocs(
    modules=[ProjectionPursuit],
    authors="Yijun Xie",
    repo="https://github.com/xieyj17/ProjectionPursuit.jl/blob/{commit}{path}#{line}",
    sitename="ProjectionPursuit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xieyj17.github.io/ProjectionPursuit.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
deploydocs(
    repo = "github.com/xieyj17/ProjectionPursuit.jl.git"
)