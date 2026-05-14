module MFTRun
    export SolveMFT!

    using FixedPointToolkit, Distributions, TightBindingToolkit, Logging, LinearAlgebra, JLD2

    using ..MeanFieldToolkit.TBMFT: TBMFTModel
    using ..MeanFieldToolkit.BdGMFT: BdGMFTModel
    using ..MeanFieldToolkit.MFTIter: MFTIterator


    function extract_data!(mft::TBMFTModel, selfcons::SelfCons, fileName::String)

        data = Dict{String, Any}(
            "Iterations" => length(selfcons.VIns) - 1,
            "MFT_Energy" => mft.MFTEnergy[end],
            "Hopping_Order" => last.(getproperty.(mft.HoppingOrders, :value)),
            "UC" => mft.model.uc,
            "Gap" => mft.model.gap,
            "mu" => mft.model.mu,
            "Outputs" => selfcons.VOuts[end],
            "Convergence" => norm(selfcons.VIns[end] - selfcons.VOuts[end])
        )

        save(fileName, data)
    end

    function extract_data!(mft::BdGMFTModel, selfcons::SelfCons, fileName::String)

        data = Dict{String, Any}(
            "Iterations" => length(selfcons.VIns) - 1,
            "MFT_Energy" => mft.MFTEnergy[end],
            "Hopping_Order" => last.(getproperty.(mft.HoppingOrders, :value)),
            "Pairing_Order" => last.(getproperty.(mft.PairingOrders, :value)),
            "UC_hopp" => mft.model.uc_hop,
            "Gap" => mft.model.gap,
            "mu" => mft.model.mu,
            "Outputs" => selfcons.VOuts[end],
            "Convergence" => norm(selfcons.VIns[end] - selfcons.VOuts[end])
        )

        save(fileName, data)
    end


@doc """
```julia
SolveMFT!(mft::TBMFTModel{T, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons
SolveMFT!(mft::BdGMFTModel{T, R, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::TBMFTModel{T, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::BdGMFTModel{T, R, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::TBMFTModel{T, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) --> SelfCons
SolveMFT!(mft::BdGMFTModel{T, R, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) --> SelfCons 
SolveMFT!(mft::TBMFTModel{T, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false) --> SelfCons
SolveMFT!(mft::BdGMFTModel{T, R, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false) --> SelfCons
```
Solves the mean-field theory on the given `MFT` object, and returns the `SelfCons` object (Refer to [FixedPointToolkit](https://github.com/Anjishnubose/FixedPointToolkit.jl)) containing the results of the mean-field theory.
- If `fileName` is passed and `debug = true`, then the `SelfCons` object is checkpointed to the file after every `checkpoint_interval` iterations.
- If `fileName` is passed and `debug = false`, then checkpointing is suppressed and only a compact final output is saved to `fileName`.
- If `Initial` is passed, then the initial order parameters are set to the values in `Initial`.
- If `Initial_range` is passed, then the initial order parameters are set to random values in the range `Initial_range`.
- If `Update` is passed, then the update function is used to perform the self-consistency update.
- If `Update_kwargs` is passed, then the keyword arguments are passed to the update function.
- If `max_iter` is passed, then the maximum number of iterations is set to `max_iter`.
- If `tol` is passed, then the tolerance for convergence is set to `tol`.

"""
    function SolveMFT!(mft::TBMFTModel{T, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFTModel{T, R, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders) + length(mft.PairingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


    function SolveMFT!(mft::TBMFTModel{T, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval, save_checkpoints = debug)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        if !debug
            extract_data!(mft, selfcons, fileName)
        end
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFTModel{T, R, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders) + length(mft.PairingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval, save_checkpoints = debug)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        if !debug
            extract_data!(mft, selfcons, fileName)
        end
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


    function SolveMFT!(mft::TBMFTModel{T, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFTModel{T, R, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::TBMFTModel{T, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval, save_checkpoints = debug)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        if !debug
            extract_data!(mft, selfcons, fileName)
        end
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFTModel{T, R, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, debug::Bool = false) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval, save_checkpoints = debug)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        if !debug
            extract_data!(mft, selfcons, fileName)
        end
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


end