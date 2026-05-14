module Build

    export BuildFromInteractions!

    using LinearAlgebra, TightBindingToolkit, Logging

    using ..MeanFieldToolkit.MFTBonds: GetMFTBonds, GetBondDictionary
    using ..MeanFieldToolkit.TBMFT:TBMFTModel
    using ..MeanFieldToolkit.BdGMFT:BdGMFTModel


    ##### ///TODO: Add Free Hopping energies also
@doc """
```julia
BuildFromInteractions!(tbMFT::TBMFTModel{T, R}, HoppingOrderLookup::Dict{Tuple, Matrix{ComplexF64}} ; refresh::Bool = true)
BuildFromInteractions!(bdgMFT::BdGMFTModel{T, R, S} , HoppingOrderLookup::Dict{Tuple, Matrix{ComplexF64}}, PairingOrderLookup::Dict{Tuple, Matrix{ComplexF64}} ; refresh::Bool = true)
```
Builds the MFT decomposed hoppings (and pairings) given a `MFT` object and the lookup tables for the expectation values of the different order parameters.
If `refresh` is set to `true`, then the MFT bonds are deleted and rebuilt from scratch. Otherwise, the new bonds are appended to the existing bonds.

"""
    function BuildFromInteractions!(tbMFT::TBMFTModel{T, R}, HoppingOrderLookup::Dict{Tuple, Matrix{ComplexF64}} ; refresh::Bool = true) where {T, R}
        ##### Deleting all the MFT bonds if refresh
        if refresh
            labels 	=	getproperty.(tbMFT.model.uc.bonds, :label)
            deleteat!(tbMFT.model.uc.bonds, findall(startswith.(labels, "MFT =>")))
        end

        @info "Building MFT Decomposed Hamiltonian..."
        for (i, Interaction) in enumerate(tbMFT.Interactions)
            ##### Making unique labels for different interaction parameters
            labels      =   Dict{String, String}()
            for key in ["ii", "jj", "ij"]
                labels[key]     =   "$(Interaction.label) -> $(tbMFT.ChannelLabels[key])"
            end
            ##### Interaction lookup table
            lookup      =   Lookup([Interaction])
            scaling = get(tbMFT.MFTScaling, Interaction.label, tbMFT.MFTScaling)
            ##### iterating over each bond in the lookup table, getting the expectation value, and decomposing the interaction using the MFT decomposition function
            for BondKey in keys(lookup)

                Expectations        =   GetBondDictionary(HoppingOrderLookup, BondKey, tbMFT.model.uc.localDim) ##### Pass Expectation value lookup into this function
                Decomposed          =   tbMFT.MFTDecomposition[i](lookup[BondKey] , Expectations)

                MFTBonds            =   GetMFTBonds(Decomposed ; BondKey = BondKey, uc = tbMFT.model.uc, scaling = Dict{String, Any}(scaling), labels = labels)
                append!(tbMFT.model.uc.bonds, MFTBonds)

            end
        end

    end

    ##### ///TODO: Add Free Hopping and pairing energies also
    function BuildFromInteractions!(bdgMFT::BdGMFTModel{T, R, S} , HoppingOrderLookup::Dict{Tuple, Matrix{ComplexF64}}, PairingOrderLookup::Dict{Tuple, Matrix{ComplexF64}} ; refresh::Bool = true) where {T, R, S}
        ##### Deleting all the MFT bonds if refresh
        if refresh
            labels 	=	getproperty.(bdgMFT.model.uc_hop.bonds, :label)
            deleteat!(bdgMFT.model.uc_hop.bonds, findall(startswith.(labels, "MFT =>")))

            labels 	=	getproperty.(bdgMFT.model.uc_pair.bonds, :label)
            deleteat!(bdgMFT.model.uc_pair.bonds, findall(startswith.(labels, "MFT =>")))
        end

        @info "Building MFT Decomposed Hamiltonian..."
        for (i, Interaction) in enumerate(bdgMFT.Interactions)
            ##### Making unique labels for different interaction parameters
            HoppingLabels      =   Dict{String, String}()
            PairingLabels      =   Dict{String, String}()
            for key in ["ii", "jj", "ij"]
                HoppingLabels[key]     =   "$(Interaction.label) -> $(bdgMFT.HoppingLabels[key])"
                PairingLabels[key]     =   "$(Interaction.label) -> $(bdgMFT.PairingLabels[key])"
            end
            ##### Interaction lookup table
            lookup      =   Lookup([Interaction])
            ##### iterating over each bond in the lookup table, getting the expectation values (hopping and pairing), and decomposing the interaction using the MFT decomposition function
            for BondKey in keys(lookup)

                HoppingExpectations =   GetBondDictionary(HoppingOrderLookup, BondKey, bdgMFT.model.uc_hop.localDim)
                PairingExpectations =   GetBondDictionary(PairingOrderLookup, BondKey, bdgMFT.model.uc_pair.localDim)

                DecomposedHopping   =   bdgMFT.HoppingDecomposition[i](lookup[BondKey] , HoppingExpectations)
                DecomposedPairing   =   bdgMFT.PairingDecomposition[i](lookup[BondKey] , PairingExpectations)

                HoppingBonds        =   GetMFTBonds(DecomposedHopping ; BondKey = BondKey, uc = bdgMFT.model.uc_hop, scaling = bdgMFT.HoppingScaling, labels = HoppingLabels)
                append!(bdgMFT.model.uc_hop.bonds, HoppingBonds)

                PairingBonds        =   GetMFTBonds(DecomposedPairing ; BondKey = BondKey, uc = bdgMFT.model.uc_pair, scaling = bdgMFT.PairingScaling, labels = PairingLabels)
                append!(bdgMFT.model.uc_pair.bonds, PairingBonds)

            end

        end

    end



end
