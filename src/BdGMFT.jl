module BdGMFT

    export BdGMFTModel, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation

    import ..MeanFieldToolkit.TBMFT: GetMFTEnergy
    

@doc """
`BdGMFTModel{T, R, S}` is a data type representing a general mean-field simulation on a BdG model.

# Attributes
- `model              ::  Model`: The BdG model on which mean-field simulations are going to run, contains info about free hopping and pairing.
- `HoppingOrders      ::  Vector{Param{2, R}}`: a vector of hopping order parameters to decompose the interactions in during MFT.
- `PairingOrders      ::  Vector{Param{2, R}}`: a vector of pairing order parameters to decompose the interactions in during MFT.
- `Interactions       ::  Vector{Param{2, FLoat64}}`: the vector of `Param` containing all the information of the interactions acting on the model.
- `HoppingDecomposition   ::  Vector{Function}` : the decomposition function which describes how to take an interaction array + hopping expectation values and give back tight-binding hoppings.
- `HoppingDecomposition   ::  Vector{Function}` : the decomposition function which describes how to take an interaction array + pairing expectation values and give back BdG pairings.
- `HoppingScaling     ::  Dict{String, Float64}`: relative scaling parameters for different hopping mean-field channels.
- `PairingScaling     ::  Dict{String, Float64}`: relative scaling parameters for different pairing mean-field channels.
- `HoppingLabels      ::  Dict{String, String}`: The labels of the different hopping mean-field channels.
- `PairingLabels      ::  Dict{String, String}`: The labels of the different hopping mean-field channels.

Initialize this structure using 
```julia
BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site"))
BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site"))
BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site"))
BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site"))
```
"""
    mutable struct BdGMFTModel{T, R, S} 
        ##### The full BdG MFT Model
        model                   ::  BdGModel
        ##### Two vectors of Params tracking hopping and pairing order parameters.
        HoppingOrders           ::  Vector{Param{2, R}}
        PairingOrders           ::  Vector{Param{2, S}}
        ##### The vector of param tracking information about the interactions, and the corresponding MFT decomposition functions for hopping and pairing
        Interactions            ::  Vector{Param{T, Float64}}
        HoppingDecomposition    ::  Vector{Function}
        PairingDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy               ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels for hopping and pairing
        HoppingScaling          ::  Dict{String, Float64}
        PairingScaling          ::  Dict{String, Float64}
        ##### The user defined labels of the different MFT channels for hopping and pairing
        HoppingLabels           ::  Dict{String, String}
        PairingLabels           ::  Dict{String, String}

        ##### TODO : Add a method which takes in a single decomposition function
        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

    end

    ##### /// TODO: Add Free Hopping and pairing energies also
    ##### TODO: Test
@doc """
```julia
GetMFTEnergy(bdgMFT::BdGMFTModel{T, R}) --> Float64
```
Returns the total mean-field energy of the BdG model including decomposed interactions.

"""
    function GetMFTEnergy(bdgMFT::BdGMFTModel{T, R, S}) :: Float64 where {T, R, S}

        Energy             =   0.0
        HoppingLookup      =   Lookup(bdgMFT.model.uc_hop)
        PairingLookup      =   Lookup(bdgMFT.model.uc_pair)

        for BondKey in keys(HoppingLookup)

            G_ij        =   GetBondCoorelation(bdgMFT.model.Gr, BondKey..., bdgMFT.model.uc_hop, bdgMFT.model.bz)
            t_ij        =   HoppingLookup[BondKey]

            Energy      +=  sum((t_ij .* G_ij))
        end

        for BondKey in keys(PairingLookup)

            F_ij        =   GetBondCoorelation(bdgMFT.model.Fr, BondKey..., bdgMFT.model.uc_pair, bdgMFT.model.bz)
            p_ij        =   PairingLookup[BondKey]

            Energy      +=  sum((p_ij .* F_ij))
        end

        return real(Energy) / length(bdgMFT.model.uc_hop.basis)
    end










end
