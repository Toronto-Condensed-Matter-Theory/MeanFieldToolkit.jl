module MFTIter
    export DecomposeGr, MFTIterator

    using TightBindingToolkit, LinearAlgebra, Statistics

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.TBMFT: TBMFTModel, GetMFTEnergy
    using ..MeanFieldToolkit.BdGMFT: BdGMFTModel, GetMFTEnergy
    using ..MeanFieldToolkit.Build: BuildFromInteractions!

    #####/// TODO : Try to vectorize maybe? Literally no need
@doc """
```julia
DecomposeGr(Gr::Array{Matrix{ComplexF64}, T}, param::Param{2, R}, uc::UnitCell{T}, bz::BZ)
```
Decomposes the Green's function into an order parameter. 
The order parameter is calculated by taking the trace decomposition of the Green's function on each bond the order parameter exists in, and then taking the mean of all the bond order parameters, following the pattern of the given order parameter.

"""
    function DecomposeGr(Gr::Array{Matrix{ComplexF64}, T}, param::Param{2, R}, uc::UnitCell{T}, bz::BZ) :: R where {T, R <: Union{Float64, ComplexF64}}

        strengths   =   R[] 

        for bond in param.unitBonds
            ##### get Greens function on the bond
            G           =   GetBondCoorelation(Gr, bond, uc, bz)
            ##### decompose the Greens function on the bond
            decomposition   =   (tr( adjoint(bond.mat) * G) / (tr(adjoint(bond.mat) * bond.mat)))
            strength        =   (R == Float64 ? real(decomposition) : decomposition)

            push!(strengths, strength)
        end
        ##### return the mean of the bond order parameters
        return mean(strengths)
    end


    ##### ///TODO Pass in the initial chemical potential guess to MFTIterator
@doc """
```julia
MFTIterator(Strengths::Vector{R}, tbMFT::TBMFTModel{T, R}) --> Vector{R}
MFTIterator(Strengths::Vector{R}, bdgMFT::BdGMFTModel{T, R, R}) --> Vector{R}
```
Runs a single iteration of the mean-field theory on the given `MFT` object, and returns the new order parameters.

"""
    function MFTIterator(Strengths::Vector{R}, tbMFT::TBMFTModel{T, R}) :: Vector{R} where {T, R <: Union{Float64, ComplexF64}}
        ##### Push the new order parameters into the `HoppingOrders` attribute of the `MFT` object
        push!.( getproperty.(tbMFT.HoppingOrders, :value) , Strengths)
        ##### Recalculate the lookup table for the new order parameters
        HoppingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(tbMFT.HoppingOrders))
        ##### Build the new MFT decomposed Hamiltonian
        BuildFromInteractions!(tbMFT, HoppingOrderLookup)
        ##### Diagonalize the Hamiltonian
        H       =   Hamiltonian(tbMFT.model.uc, tbMFT.model.bz)
        DiagonalizeHamiltonian!(H)
        ##### Pass the Hamiltonian to the `Model` object
        tbMFT.model.Ham     =   H
        ##### ///TODOPass initial chemical potential guess to SolveModel!
        ##### Solve the model, taking in the initial chemical potential guess as the previous chemical potential.
        SolveModel!(tbMFT.model ; mu_guess = tbMFT.model.mu)
        ##### Push the new mean-field energy into the `MFTEnergy` attribute of the `MFT` object
        push!(tbMFT.MFTEnergy, GetMFTEnergy(tbMFT))
        ##### Decompose the new Green's function into the new order parameters
        NewStrengths    =   DecomposeGr.(Ref(tbMFT.model.Gr), tbMFT.HoppingOrders, Ref(tbMFT.model.uc), Ref(tbMFT.model.bz))

        return NewStrengths
    end

    function MFTIterator(Strengths::Vector{R}, bdgMFT::BdGMFTModel{T, R, R}) :: Vector{R} where {T, R <: Union{Float64, ComplexF64}}
        ##### Push the new order parameters into the `HoppingOrders` and `PairingOrders` attributes of the `MFT` object
        push!.( getproperty.(bdgMFT.HoppingOrders, :value) , Strengths[begin : length(bdgMFT.HoppingOrders)])
        push!.( getproperty.(bdgMFT.PairingOrders, :value) , Strengths[length(bdgMFT.HoppingOrders) + 1 : end])
        ##### Recalculate the lookup table for the new order parameters
        HoppingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(bdgMFT.HoppingOrders))
        PairingOrderLookup =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(bdgMFT.PairingOrders))
        ##### Build the new MFT decomposed Hamiltonian
        BuildFromInteractions!(bdgMFT, HoppingOrderLookup, PairingOrderLookup)
        ##### Diagonalize the Hamiltonian
        H       =   Hamiltonian(bdgMFT.model.uc_hop, bdgMFT.model.uc_pair, bdgMFT.model.bz)
        DiagonalizeHamiltonian!(H)
        ##### Pass the Hamiltonian to the `Model` object
        bdgMFT.model.Ham     =   H
        ##### Solve the model, taking in the initial chemical potential guess as the previous chemical potential.
        SolveModel!(bdgMFT.model ; mu_guess = bdgMFT.model.mu)
        ##### Push the new mean-field energy into the `MFTEnergy` attribute of the `MFT` object
        push!(bdgMFT.MFTEnergy, GetMFTEnergy(bdgMFT))
        ##### Decompose the new Green's function into the new order parameters
        NewStrengths    =   DecomposeGr.(Ref(bdgMFT.model.Gr), bdgMFT.HoppingOrders, Ref(bdgMFT.model.uc_hop), Ref(bdgMFT.model.bz))
        append!(NewStrengths, DecomposeGr.(Ref(bdgMFT.model.Fr), bdgMFT.PairingOrders, Ref(bdgMFT.model.uc_pair), Ref(bdgMFT.model.bz)))

        return NewStrengths
    end








end