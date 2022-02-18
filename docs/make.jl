using ProjectionPursuit
using Documenter

DocMeta.setdocmeta!(ProjectionPursuit, :DocTestSetup, :(using ProjectionPursuit); recursive=true)

makedocs(;
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

deploydocs(;
    repo="github.com/xieyj17/ProjectionPursuit.jl.git",
    devbranch="main",
)
