module MFTBonds
    export GetBondCoorelation, GetMFTBonds, GetBondDictionary

    using TightBindingToolkit, LinearAlgebra, Tullio


@doc """
```julia
GetBondCoorelation(Gr::Array{Matrix{ComplexF64}, T}, base::Int64, target::Int64, offset::Vector{Int64}, uc::UnitCell, bz::BZ) --> Matrix{ComplexF64}
GetBondCoorelation(Gr::Array{Matrix{ComplexF64}, T}, bond::Bond, uc::UnitCell, bz::BZ) --> Matrix{ComplexF64}
```
Returns the Greens function (correlations) on the given bond from the full Array in `Gr`.

"""
    function GetBondCoorelation(Gr::Array{Matrix{ComplexF64}, T}, base::Int64, target::Int64, offset::Vector{Int64}, uc::UnitCell, bz::BZ) :: Matrix{ComplexF64} where {T}
        index       =   mod.((-offset) , bz.gridSize) .+ ones(Int64, length(offset))
        ##### TODO : the extra - sign in offset is because right now G[r] = <f^{dagger}_0 . f_{-r}> ===> NEED TO FIX THIS
        b1          =   uc.localDim * (base   - 1) + 1
        b2          =   uc.localDim * (target - 1) + 1

        G           =   Gr[index...][b1 : b1 + uc.localDim - 1, b2 : b2 + uc.localDim - 1]
        return G
    end

    function GetBondCoorelation(Gr::Array{Matrix{ComplexF64}, T}, bond::Bond, uc::UnitCell, bz::BZ) :: Matrix{ComplexF64} where {T}
        index       =   mod.((-bond.offset) , bz.gridSize) .+ ones(Int64, length(bond.offset))
        ##### TODO : the extra - sign in offset is because right now G[r] = <f^{dagger}_0 . f_{-r}> ===> NEED TO FIX THIS
        b1          =   uc.localDim * (bond.base   - 1) + 1
        b2          =   uc.localDim * (bond.target - 1) + 1

        G           =   Gr[index...][b1 : b1 + uc.localDim - 1, b2 : b2 + uc.localDim - 1]
        return G
    end


@doc """
```julia
GetBondDictionary(BondLookup::Dict{Tuple, Matrix{ComplexF64}}, BondKey::Tuple{Int64, Int64, Vector{Int64}}, localDim::Int64) --> Dict{String, Matrix{ComplexF64}}
```
Given a lookup dictionary `BondLookup`, and a bond with `BondKey=(i, j, offset)`, returns a dictionary containing the effective on-site matrices as well the bond matrices on sites `i, j` and bond `i->j`.

"""
    function GetBondDictionary(BondLookup::Dict{Tuple, Matrix{ComplexF64}}, BondKey::Tuple{Int64, Int64, Vector{Int64}}, localDim::Int64) :: Dict{String, Matrix{ComplexF64}}
        base, target, offset    =   BondKey
        AdjBondKey              =   (target, base, -offset)

        if BondKey != AdjBondKey
            Expectation_ij      =   (         get(BondLookup,    BondKey, zeros(ComplexF64, repeat([localDim], 2)...))
                                    + adjoint(get(BondLookup, AdjBondKey, zeros(ComplexF64, repeat([localDim], 2)...))))

        else
            Expectation_ij      =   (         get(BondLookup,    BondKey, zeros(ComplexF64, repeat([localDim], 2)...))
                                    + adjoint(get(BondLookup, AdjBondKey, zeros(ComplexF64, repeat([localDim], 2)...)))) / 2
        end

        Expectation_ii      =   get(BondLookup, (  base,   base, zeros(Int64, length(offset))), zeros(ComplexF64, repeat([localDim], 2)...))
        Expectation_jj      =   get(BondLookup, (target, target, zeros(Int64, length(offset))), zeros(ComplexF64, repeat([localDim], 2)...))

        Expectations        =   Dict{String, Matrix{ComplexF64}}("ij" => Expectation_ij, "ii" => Expectation_ii, "jj" => Expectation_jj)

        return Expectations

    end


@doc """
```julia
GetMFTBonds(DecomposedBonds::Dict{String, Matrix{ComplexF64}} ; BondKey::Tuple{Int64, Int64, Vector{Int64}}, uc::UnitCell{2}, scaling::Dict{String, Float64} = Dict("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), labels::Dict{String, String} = Dict("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
```
Returns a vector of `Bond` objects using the bond dictionary created by [`GetBondDictionary`][@ref]. The bond objects have `labels`, and are scaled according to `scaling`.

"""
    function GetMFTBonds(DecomposedBonds::Dict{String, Matrix{ComplexF64}} ; BondKey::Tuple{Int64, Int64, Vector{Int64}}, uc::UnitCell{2}, scaling::Dict{String, Float64} = Dict("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), labels::Dict{String, String} = Dict("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) :: Vector{Bond{2}}

        base, target, offset = BondKey

        t_ij        =   get(DecomposedBonds, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_ii        =   get(DecomposedBonds, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_jj        =   get(DecomposedBonds, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        t_ijBond    =   Bond(BondKey..., scaling["ij"] * t_ij, GetDistance(uc, BondKey...), "MFT => $(labels["ij"]) : $(string(BondKey))")

        t_iiBond    =   Bond(  base,   base, zeros(Int64, length(offset)), scaling["ii"] * t_ii, 0.0, "MFT => $(labels["ii"]) : $(string(BondKey))")
        t_jjBond    =   Bond(target, target, zeros(Int64, length(offset)), scaling["jj"] * t_jj, 0.0, "MFT => $(labels["jj"]) : $(string(BondKey))")

        return [t_ijBond, t_iiBond, t_jjBond]

    end



end
