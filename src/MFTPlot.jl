module MFTPlot
    export PlotMFT!, PlotMFTEnergy!

    using Plots

    using ..MeanFieldToolkit.TBMFT: TBMFTModel
    using ..MeanFieldToolkit.BdGMFT: BdGMFTModel


@doc """
```julia
PlotMFT!(mft::TBMFTModel ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)
PlotMFT!(mft::BdGMFTModel ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)
```
Plots the order parameters of the given `MFT` object.
- If `plot_labels` is passed, then only the order parameters with the given labels are plotted.
- If `plot_legend` is passed, then the legend is shown in the plot.
"""
    function PlotMFT!(mft::TBMFTModel ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)

        p = plot(grid=false, legend = plot_legend, bg_legend = :transparent, framestyle = :box)

        for param in mft.HoppingOrders
            if param.label in plot_labels

                plot!(param.value, marker = :circle, lw = 2.0, label = param.label)
            end
        end

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Parameters", guidefontsize = 9)
        title!("MFT parameters", titlefontsize = 12)

        return p
    end


    function PlotMFT!(mft::BdGMFTModel ; plot_labels::Vector{String} = getproperty.(mft.HoppingBlock.params, :label), plot_legend::Bool = true)

        p = plot(grid=false, legend = plot_legend, bg_legend = :transparent, framestyle = :box)

        for param in mft.HoppingOrders
            if param.label in plot_labels

                plot!(param.value, marker = :circle, lw = 2.0, label = param.label)
            end
        end

        for param in mft.PairingOrders
            if param.label in plot_labels

                plot!(param.value, marker = :square, lw = 2.0, label = param.label)
            end
        end

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Parameters", guidefontsize = 9)
        title!("BdG MFT parameters", titlefontsize = 12)

        return p
    end


@doc """
```julia
PlotMFTEnergy!(mft::T) where {T<:Union{TBMFTModel, BdGMFTModel}}
```
Plots the mean-field energy of the given `MFT` object.
"""
    function PlotMFTEnergy!(mft::T) where {T<:Union{TBMFTModel, BdGMFTModel}}

        plot!(mft.MFTEnergy, marker = :circle, lw = 2.0, label = "MFT Energy")

        xlabel!("Iterations", guidefontsize = 9)
        ylabel!("Energy", guidefontsize = 9)

        return p
    end





































end