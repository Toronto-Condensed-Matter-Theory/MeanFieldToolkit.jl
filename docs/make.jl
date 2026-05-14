using Documenter
using MeanFieldToolkit

makedocs(
    build       =   "build" ,
    sitename    =   "MeanFieldToolkit.jl"    ,
    modules     =   [MeanFieldToolkit.MFTDecompose, MeanFieldToolkit.MFTBonds, MeanFieldToolkit.TBMFT, MeanFieldToolkit.BdGMFT, MeanFieldToolkit.Build, MeanFieldToolkit.MFTIter, MeanFieldToolkit.MFTRun, MeanFieldToolkit.MFTResume, MeanFieldToolkit.MFTPlot, MeanFieldToolkit.InteractionConvert]   ,
    pages = [
        "Introduction"              =>  "index.md",
        "MFTDecompose"              =>  "MFTDecompose.md",
        "MFTBonds"                  =>  "MFTBonds.md",
        "TightBindingMFT"           =>  "TightBindingMFT.md",
        "BdGMFT"                    =>  "BdGMFT.md",
        "Build"                     =>  "Build.md",
        "MFTIterator"               =>  "MFTIterator.md",
        "MFTRun"                    =>  "MFTRun.md",
        "MFTResume"                 =>  "MFTResume.md",
        "MFTPlot"                   =>  "MFTPlot.md",
        "InteractionConvert"        =>  "InteractionConvert.md"
    ]
)

deploydocs(
    repo = "github.com/andrewkhardy/MeanFieldToolkit.jl.git",
    devbranch = "main"
)