module TBMFT

    export TBMFTModel, GetMFTEnergy, GetMFTEntropy, GetMFTFreeEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation, GetBondDictionary
    using ..MeanFieldToolkit.MFTEnergies: GetMFTBondEnergies, BondMultiplicity, FermionEntropy
    using ..MeanFieldToolkit.MFTDecompose: CheckDecompositions


@doc """
`TBMFTModel{T, R}` is a data type representing a general mean-field simulation on a tight-binding model.

# Attributes
- `model              ::  Model`: The tight-binding model on which mean-field simulations are going to run, contains info about free hoppings.
- `HoppingOrders      ::  Vector{Param{2, R}}`: a vector of order parameters to decompose the interactions in during MFT.
- `Interactions       ::  Vector{Param{2, FLoat64}}`: the vector of `Param` containing all the information of the interactions acting on the model.
- `MFTDecomposition   ::  Vector{Function}` : the decomposition function which describes how to take an interaction array + expectation values and give back tight-binding hoppings.
- `MFTScaling         ::  Dict{String, Float64}`: relative scaling parameters for different mean-field channels.
- `ChannelLabels      ::  Dict{String, String}`: The labels of the different mean-field channels.

Initialize this structure using
```julia
TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
```
"""
    mutable struct TBMFTModel{T, R}
        ##### TightBinding Model for MFT
        model               ::  Model
        ##### Vector of Params containing the expected order parameters of MFT
        HoppingOrders       ::  Vector{Param{2, R}}
        ##### Vector of Params tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions
        Interactions        ::  Vector{Param{T, Float64}}
        MFTDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy           ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels
        MFTScaling          ::  Dict{String, Float64}
        ##### The user defined labels of the different MFT channels
        ChannelLabels       ::  Dict{String, String}


        function TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}} , MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @debug "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, MFTDecomposition)

            return new{T, R}(model, HoppingOrders, Interactions, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            CheckDecompositions(Interactions, MFTDecomposition)

            return new{T, R}(model, HoppingOrders, Interactions, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}} , MFTDecomposition::Function ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @debug "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, MFTDecomposition)

            return new{T, R}(model, HoppingOrders, Interactions, repeat(Function[MFTDecomposition], length(Interactions)), Float64[], MFTScaling, ChannelLabels)
        end

        function TBMFTModel(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Function, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            CheckDecompositions(Interactions, MFTDecomposition)

            return new{T, R}(model, HoppingOrders, Interactions, repeat(Function[MFTDecomposition], length(Interactions)), Float64[], MFTScaling, ChannelLabels)
        end

    end


@doc """
```julia
GetMFTEnergy(mft::TBMFTModel{T, R}) --> Float64
```
Returns the total mean-field energy of the model including decomposed interactions.

"""
    function GetMFTEnergy(mft::TBMFTModel{T, R}) :: Float64 where {T, R}

        Energy      =   0.0
        lookup      =   Lookup(mft.model.uc)

        for BondKey in keys(lookup)

            G_ij        =   GetBondCoorelation(mft.model.Gr, BondKey..., mft.model.uc, mft.model.bz)

            t_ij        =   lookup[BondKey]
            ##### inter-site bonds enter the Hamiltonian together with their Hermitian conjugate
            Energy      +=  BondMultiplicity(BondKey) * real(sum((t_ij .* G_ij)))
        end

        DC_Energy   =   0.0
        HoppingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(mft.HoppingOrders))

        for (i, Interaction) in enumerate(mft.Interactions)
            IntLookup      =   Lookup([Interaction])
            scaling = mft.MFTScaling

            for BondKey in keys(IntLookup)
                Expectations        =   GetBondDictionary(HoppingOrderLookup, BondKey, mft.model.uc.localDim)
                Decomposed          =   mft.MFTDecomposition[i](IntLookup[BondKey] , Expectations)

                ##### GetMFTBondEnergies already returns the double-counting-corrected interaction energy (its internal /2 on "ii"/"jj" is the Euler-theorem halving); dividing again here double-corrects.
                DC_Energy += GetMFTBondEnergies(Expectations, Decomposed, mft.model.uc ; scaling = scaling)
            end
        end

        return real(Energy - DC_Energy) / length(mft.model.uc.basis)
    end


@doc """
```julia
GetMFTEntropy(mft::TBMFTModel) --> Float64
```
Returns the entropy per site (in units of k_B) of the mean-field quasiparticles at the model's temperature,
S = Σ_{k,n} s(f(ε_kn - μ)) / N_sites with s the [`FermionEntropy`](@ref). Zero at `T = 0`.

"""
    function GetMFTEntropy(mft::TBMFTModel) :: Float64
        M   =   mft.model
        @assert M.stat == -1 "GetMFTEntropy is implemented for fermions only."
        iszero(M.T) && return 0.0

        S   =   sum(sum(E -> FermionEntropy((E - M.mu) / M.T), bands) for bands in M.Ham.bands)
        return S / (length(M.Ham.bands) * length(M.uc.basis))
    end


@doc """
```julia
GetMFTFreeEnergy(mft::TBMFTModel) --> Float64
```
Returns the mean-field (Helmholtz) free energy per site at fixed filling, F = E - T S, with E from [`GetMFTEnergy`](@ref)
and S from [`GetMFTEntropy`](@ref). At self-consistency F is stationary in the order parameters, so it is the quantity
to compare between competing phases at finite temperature (at T → 0 it reduces to E).

"""
    GetMFTFreeEnergy(mft::TBMFTModel) :: Float64 = GetMFTEnergy(mft) - mft.model.T * GetMFTEntropy(mft)





end
