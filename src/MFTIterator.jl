module MFTIter
    export DecomposeGr, MFTIterator, SolveFilling!

    using TightBindingToolkit, LinearAlgebra, Statistics

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.TBMFT: TBMFTModel, GetMFTEnergy
    using ..MeanFieldToolkit.BdGMFT: BdGMFTModel, GetMFTEnergy
    using ..MeanFieldToolkit.Build: BuildFromInteractions!

    #####/// TODO : Try to vectorize maybe? Literally no need
@doc """
```julia
DecomposeGr(Gr::Array{Matrix{ComplexF64}, N}, param::Param{2, R}, uc::UnitCell{T}, bz::BZ)
```
Decomposes the Green's function into an order parameter. 
The order parameter is calculated by taking the trace decomposition of the Green's function on each bond the order parameter exists in, and then taking the mean of all the bond order parameters, following the pattern of the given order parameter.

"""
    function DecomposeGr(Gr::Array{Matrix{ComplexF64}, N}, param::Param{2, R}, uc::UnitCell{T}, bz::BZ) :: R where {N,T, R <: Union{Float64, ComplexF64}}

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


    ##### The chemical potential that gives the `target` filling, inside a bracket [lo, hi] that always contains it.
    ##### (TightBindingToolkit's GetMu! searches 10 .* bandwidth, which misses μ when every band lies on one side
    #####  of zero, e.g. after a large Hartree shift, and then silently fills every level.)
    ##### Starts from `guess` (the previous μ, usually almost right during a self-consistency loop), expands outward until
    ##### the root is bracketed, then refines with Illinois regula falsi, which converges superlinearly and never leaves the bracket.
    function FindMu(filling::Function, target::Float64, lo::Float64, hi::Float64, guess::Float64 ; mu_tol::Float64, filling_tol::Float64) :: Float64
        g(mu)   =   filling(mu) - target          ##### increasing in mu
        x       =   clamp(guess, lo, hi)
        gx      =   g(x)
        abs(gx) < filling_tol && return x

        ##### bracket [a, b] with g(a) < 0 < g(b), growing outward from the guess
        step    =   1e-4
        if gx > 0
            b, gb   =   x, gx
            a       =   max(x - step, lo) ; ga = g(a)
            while ga > 0 && a > lo
                b, gb   =   a, ga
                step    *=  8
                a       =   max(x - step, lo) ; ga = g(a)
            end
        else
            a, ga   =   x, gx
            b       =   min(x + step, hi) ; gb = g(b)
            while gb < 0 && b < hi
                a, ga   =   b, gb
                step    *=  8
                b       =   min(x + step, hi) ; gb = g(b)
            end
        end

        side    =   0
        for _ in 1:500
            abs(ga) < filling_tol && return a
            abs(gb) < filling_tol && return b
            b - a < mu_tol && return (a + b) / 2
            c       =   (a * gb - b * ga) / (gb - ga)
            c       =   a < c < b ? c : (a + b) / 2
            gc      =   g(c)
            abs(gc) < filling_tol && return c
            if gc < 0
                a, ga   =   c, gc
                side == -1 && (gb /= 2)
                side    =   -1
            else
                b, gb   =   c, gc
                side == 1 && (ga /= 2)
                side    =   1
            end
        end
        return (a + b) / 2
    end

@doc """
```julia
SolveFilling!(M::Model ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10)
SolveFilling!(M::BdGModel ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10)
```
Finds the chemical potential for the model's filling and computes the correlations `Gr` (and `Fr` for BdG), like
TightBindingToolkit's `SolveModel!`, but searching a bracket that always contains the chemical potential (starting from the previous μ).
At `T = 0`, or when the model was given a chemical potential instead of a filling, it calls `SolveModel!`.

"""
    function SolveFilling!(M::Model ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10)
        if M.filling < 0 || iszero(M.T)
            SolveModel!(M ; mu_guess = M.mu, mu_tol = mu_tol, filling_tol = filling_tol, verbose = false)
            return
        end
        energies    =   reduce(vcat, M.Ham.bands)
        margin      =   1.0 + 40 * M.T
        M.mu        =   FindMu(mu -> FindFilling(mu, M.Ham.bands, M.T, M.stat), M.filling, minimum(energies) - margin, maximum(energies) + margin, M.mu ; mu_tol = mu_tol, filling_tol = filling_tol)
        GetGk!(M)
        GetGr!(M)
    end

    function SolveFilling!(M::BdGModel ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10)
        if M.filling < 0
            SolveModel!(M ; mu_guess = M.mu, mu_tol = mu_tol, filling_tol = filling_tol, verbose = false)
            return
        end
        ##### the BdG bands are ±|ε - μ'| for the field μ' they were built with (0 for a fresh Hamiltonian), so every
        ##### normal-state level, and hence μ, lies within maximum|E| of μ'
        width       =   maximum(abs, reduce(vcat, M.Ham.bands)) + 1.0 + 40 * M.T
        M.mu        =   FindMu(mu -> FindFilling(mu, M), M.filling, min(0.0, M.mu) - width, max(0.0, M.mu) + width, M.mu ; mu_tol = mu_tol, filling_tol = filling_tol)
        FindFilling(M.mu, M)        ##### leaves the Hamiltonian at the final μ
        GetGk!(M)
        GetGr!(M)
    end


    ##### ///TODO Pass in the initial chemical potential guess to MFTIterator
@doc """
```julia
MFTIterator(Strengths::Vector{R}, tbMFT::TBMFTModel{T, R} ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10) --> Vector{R}
MFTIterator(Strengths::Vector{R}, bdgMFT::BdGMFTModel{T, R, R} ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10) --> Vector{R}
```
Runs a single iteration of the mean-field theory on the given `MFT` object, and returns the new order parameters.

`mu_tol` and `filling_tol` set how accurately the chemical potential is solved for at each iteration.
They limit how precisely the self-consistency can converge: with a loose `mu_tol` the returned order parameters are noisy at roughly that level, and the iteration stalls there.

"""
    function MFTIterator(Strengths::Vector{R}, tbMFT::TBMFTModel{T, R} ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10) :: Vector{R} where {T, R <: Union{Float64, ComplexF64}}
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
        SolveFilling!(tbMFT.model ; mu_tol = mu_tol, filling_tol = filling_tol)
        ##### Push the new mean-field energy into the `MFTEnergy` attribute of the `MFT` object
        push!(tbMFT.MFTEnergy, GetMFTEnergy(tbMFT))
        ##### Decompose the new Green's function into the new order parameters
        NewStrengths    =   DecomposeGr.(Ref(tbMFT.model.Gr), tbMFT.HoppingOrders, Ref(tbMFT.model.uc), Ref(tbMFT.model.bz))

        return NewStrengths
    end

    function MFTIterator(Strengths::Vector{R}, bdgMFT::BdGMFTModel{T, R, R} ; mu_tol::Float64 = 1e-10, filling_tol::Float64 = 1e-10) :: Vector{R} where {T, R <: Union{Float64, ComplexF64}}
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
        SolveFilling!(bdgMFT.model ; mu_tol = mu_tol, filling_tol = filling_tol)
        ##### Push the new mean-field energy into the `MFTEnergy` attribute of the `MFT` object
        push!(bdgMFT.MFTEnergy, GetMFTEnergy(bdgMFT))
        ##### Decompose the new Green's function into the new order parameters
        NewStrengths    =   DecomposeGr.(Ref(bdgMFT.model.Gr), bdgMFT.HoppingOrders, Ref(bdgMFT.model.uc_hop), Ref(bdgMFT.model.bz))
        append!(NewStrengths, DecomposeGr.(Ref(bdgMFT.model.Fr), bdgMFT.PairingOrders, Ref(bdgMFT.model.uc_pair), Ref(bdgMFT.model.bz)))

        return NewStrengths
    end








end