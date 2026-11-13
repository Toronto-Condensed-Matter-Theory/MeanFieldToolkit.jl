module BdGMFT

    export BdGMFTModel, GetMFTEnergy, GetMFTEntropy, GetMFTFreeEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation, GetBondDictionary
    using ..MeanFieldToolkit.MFTEnergies: GetMFTBondEnergies, BondMultiplicity, FermionEntropy
    using ..MeanFieldToolkit.MFTDecompose: CheckDecompositions

    import ..MeanFieldToolkit.TBMFT: GetMFTEnergy, GetMFTEntropy, GetMFTFreeEnergy
    

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

            @debug "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            @debug "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            @debug "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, HoppingOrders::Vector{Param{2, R}}, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, R, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            @debug "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

            return new{T, S, S}(model, HoppingOrders, PairingOrders, Interactions, repeat(Function[HoppingDecomposition], length(Interactions)), repeat(Function[PairingDecomposition], length(Interactions)), Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFTModel(model::BdGModel, PairingOrders::Vector{Param{2, S}}, Interactions::Vector{Param{T, Float64}} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, S <: Union{Float64, ComplexF64}}

            @warn "No Hopping Order parameters passed."
            HoppingOrders       =   Param{2, S}[]

            CheckDecompositions(Interactions, HoppingDecomposition)
            CheckDecompositions(Interactions, PairingDecomposition)

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

            Energy      +=  BondMultiplicity(BondKey) * real(sum((t_ij .* G_ij)))
        end

        for BondKey in keys(PairingLookup)

            F_ij        =   GetBondCoorelation(bdgMFT.model.Fr, BondKey..., bdgMFT.model.uc_pair, bdgMFT.model.bz)
            p_ij        =   PairingLookup[BondKey]

            ##### every pairing term, on-site included, enters the Hamiltonian together with its Hermitian conjugate
            Energy      +=  2 * real(sum((p_ij .* F_ij)))
        end

        DC_Energy = 0.0
        HoppingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(bdgMFT.HoppingOrders))
        PairingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(bdgMFT.PairingOrders))

        for (i, Interaction) in enumerate(bdgMFT.Interactions)
            IntLookup      =   Lookup([Interaction])

            for BondKey in keys(IntLookup)
                HoppingExpectations =   GetBondDictionary(HoppingOrderLookup, BondKey, bdgMFT.model.uc_hop.localDim)
                PairingExpectations =   GetBondDictionary(PairingOrderLookup, BondKey, bdgMFT.model.uc_pair.localDim)

                DecomposedHopping   =   bdgMFT.HoppingDecomposition[i](IntLookup[BondKey] , HoppingExpectations)
                DecomposedPairing   =   bdgMFT.PairingDecomposition[i](IntLookup[BondKey] , PairingExpectations)

                ##### GetMFTBondEnergies returns the double-counting energy ½⟨V_MF⟩ directly; on-site pairing terms carry a Hermitian conjugate, so they are not halved.
                DC_Energy += GetMFTBondEnergies(HoppingExpectations, DecomposedHopping, bdgMFT.model.uc_hop ; scaling = bdgMFT.HoppingScaling)
                DC_Energy += GetMFTBondEnergies(PairingExpectations, DecomposedPairing, bdgMFT.model.uc_pair ; scaling = bdgMFT.PairingScaling, onsite_weight = 1.0)
            end
        end

        return real(Energy - DC_Energy) / length(bdgMFT.model.uc_hop.basis)
    end


@doc """
```julia
GetMFTEntropy(bdgMFT::BdGMFTModel) --> Float64
```
Returns the entropy per site (in units of k_B) of the Bogoliubov quasiparticles at the model's temperature.
The BdG spectrum (with μ already included) contains every quasiparticle twice, at ±E, so only the lower half is summed:
S = Σ_{k, n ≤ N/2} s(E_kn / T) / N_sites. Zero at `T = 0`.

"""
    function GetMFTEntropy(bdgMFT::BdGMFTModel) :: Float64
        M   =   bdgMFT.model
        @assert M.stat == -1 "GetMFTEntropy is implemented for fermions only."
        iszero(M.T) && return 0.0

        half    =   length(first(M.Ham.bands)) ÷ 2
        S       =   sum(sum(E -> FermionEntropy(E / M.T), bands[1:half]) for bands in M.Ham.bands)
        return S / (length(M.Ham.bands) * length(M.uc_hop.basis))
    end

    GetMFTFreeEnergy(bdgMFT::BdGMFTModel) :: Float64 = GetMFTEnergy(bdgMFT) - bdgMFT.model.T * GetMFTEntropy(bdgMFT)

end
