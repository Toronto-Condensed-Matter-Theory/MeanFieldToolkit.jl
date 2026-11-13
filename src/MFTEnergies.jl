module MFTEnergies

    export GetMFTBondEnergies, BondMultiplicity, FermionEntropy

    using TightBindingToolkit

@doc """
```julia
FermionEntropy(x::Float64) --> Float64
```
Entropy of one fermionic mode with energy `x = (E - μ) / T` (in units of k_B):
s = -f ln f - (1 - f) ln(1 - f) with f = 1 / (eˣ + 1), evaluated in the overflow-free form ln(1 + e^{-|x|}) + |x| / (e^{|x|} + 1).

"""
    function FermionEntropy(x::Float64) :: Float64
        y   =   abs(x)
        return log1p(exp(-y)) + y / (exp(y) + 1)
    end

@doc """
```julia
BondMultiplicity(BondKey::Tuple{Int64, Int64, Vector{Int64}}) --> Int64
```
How many times a bond in a `Lookup` table enters the Hamiltonian: once for an on-site term `(i, i, 0)`, and twice
(the bond and its Hermitian conjugate) for every other bond. `Lookup` merges a bond with its reverse into one entry,
so energies summed over a lookup table must weight each entry by this factor.

"""
    BondMultiplicity(BondKey::Tuple) :: Int64 = (BondKey[1] == BondKey[2] && iszero(BondKey[3])) ? 1 : 2

    #### /// TODO : Add relative scaling to energy calculation   -----> Bonds are already scaled, so dont need to do this!!!
    
@doc """
```julia
GetMFTBondEnergies(Chis::Dict{String, Matrix{ComplexF64}}, DecomposedBonds::Dict{String, Matrix{ComplexF64}}, uc::UnitCell ; scaling :: Dict{String, Float64} = Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), onsite_weight::Float64 = 0.5) --> Float64
```
Returns the double-counting energy ½⟨V_MF⟩ of one decomposed interaction bond, given the expectation values `Chis`.
Everything is re-scaled according to `scaling`.

A bond term ("ij") enters the mean-field Hamiltonian together with its Hermitian conjugate, so ½⟨V_MF⟩ = Re(t_ij · χ_ij).
An on-site hopping term ("ii", "jj") enters once, so it is weighted by `onsite_weight = 1/2`. On-site *pairing* terms also come
with their Hermitian conjugate, so pairing energies use `onsite_weight = 1`.

"""
    function GetMFTBondEnergies(Chis::Dict{String, Matrix{ComplexF64}}, DecomposedBonds::Dict{String, Matrix{ComplexF64}}, uc::UnitCell ; scaling :: Dict{String, Float64} = Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), onsite_weight::Float64 = 0.5) :: Float64

        t_ij        =   get(DecomposedBonds, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_ii        =   get(DecomposedBonds, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_jj        =   get(DecomposedBonds, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        chi_ij      =   get(Chis, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        chi_ii      =   get(Chis, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        chi_jj      =   get(Chis, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        energy      =   sum(scaling["ij"] .* (t_ij .* chi_ij) + onsite_weight .* (scaling["ii"] .* (t_ii .* chi_ii) + scaling["jj"] .* (t_jj .* chi_jj)))
        return real(energy)
    end

end