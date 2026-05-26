module MFTEnergies

    export GetMFTBondEnergies

    using TightBindingToolkit
    #### /// TODO : Add relative scaling to energy calculation   -----> Bonds are already scaled, so dont need to do this!!!
    
@doc """
```julia
GetMFTBondEnergies(Chis::Dict{String, Matrix{ComplexF64}}, DecomposedBonds::Dict{String, Matrix{ComplexF64}}, uc::UnitCell ; scaling :: Dict{String, Float64} = Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0) ) --> Float64
```
Returns the mean-field energy contribution given an interaction and expectation value `Chis`. Everything is re-scaled according to `scaling`. 

"""
    function GetMFTBondEnergies(Chis::Dict{String, Matrix{ComplexF64}}, DecomposedBonds::Dict{String, Matrix{ComplexF64}}, uc::UnitCell ; scaling :: Dict{String, Float64} = Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0) ) :: Float64

        t_ij        =   get(DecomposedBonds, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_ii        =   get(DecomposedBonds, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_jj        =   get(DecomposedBonds, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        chi_ij      =   get(Chis, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        chi_ii      =   get(Chis, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        chi_jj      =   get(Chis, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        energy      =   sum(scaling["ij"] .* (t_ij .* chi_ij) + ((scaling["ii"] .* (t_ii .* chi_ii) + scaling["jj"] .* (t_jj .* chi_jj)) / 2))
        return real(energy)
    end

end