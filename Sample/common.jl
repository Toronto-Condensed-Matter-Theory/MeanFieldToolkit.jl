#=
Shared code for the notebooks in Sample/Notebooks (and for test/test_examples.jl). Everything here is a short
function wrapping exactly what the scripts in Sample/MinimalExamples spell out by hand:

- the four lattices (square, triangular, honeycomb, kagome) in the smallest unit cells that hold their natural
  magnetic orders (Néel, 120°, Néel, q = 0), nearest-neighbour distance 1;
- unrestricted order parameters: one Param per site or nearest-neighbour bond and per matrix of a complete
  2×2 basis, so every spin (and pairing) channel is free;
- builders for the Hubbard, Heisenberg (fermionic spinons) and slave-boson t-J models;
- a physically motivated starting point for SolveMFT!;
- analysis: site moments, magnetic pattern, bond hopping and pairing amplitudes, pairing symmetry.
=#

using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging

const SPIN          =   SpinMats(1 // 2)                    ##### [Sx, Sy, Sz, Id]
const ISIGMAY       =   ComplexF64[0 1; -1 0]               ##### iσʸ: spin-singlet pairing; 2 S^μ iσʸ are the triplets

##### Complete bases of 2×2 matrices, with the channel names used in the order-parameter labels
const HOPPING_CHANNELS  =   ["Sx" => SPIN[1], "Sy" => SPIN[2], "Sz" => SPIN[3], "n" => SPIN[4]]
const BOND_CHANNELS     =   ["χx" => SPIN[1], "χy" => SPIN[2], "χz" => SPIN[3], "χ" => SPIN[4]]
const PAIRING_CHANNELS  =   ["Δs" => ISIGMAY, "Δx" => 2SPIN[1] * ISIGMAY, "Δy" => 2SPIN[2] * ISIGMAY, "Δz" => 2SPIN[3] * ISIGMAY]

const HUBBARD_U     =   DensityToPartonCoupling([1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0])    ##### n↑ n↓
const HEISENBERG_J  =   SpinToPartonCoupling(Matrix{Float64}(I, 3, 3), 1 // 2)               ##### S_i · S_j


##### ------------------------------------------------- Lattices -------------------------------------------------

const LATTICES  =   (:square, :triangular, :honeycomb, :kagome)

const GEOMETRY  =   Dict(
    :square     =>  (primitives = [[1.0, 1.0], [1.0, -1.0]],                basis = [[0.0, 0.0], [1.0, 0.0]]),                          ##### √2 × √2 cell
    :triangular =>  (primitives = [[1.5, sqrt(3) / 2], [0.0, sqrt(3)]],     basis = [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2]]),      ##### √3 × √3 cell
    :honeycomb  =>  (primitives = [[sqrt(3) / 2, 1.5], [-sqrt(3) / 2, 1.5]], basis = [[0.0, 0.0], [0.0, 1.0]]),
    :kagome     =>  (primitives = [[2.0, 0.0], [1.0, sqrt(3)]],             basis = [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2]]),
)

##### Classical ground-state spin directions of the antiferromagnet, per site of the unit cell
const CLASSICAL_SPINS   =   Dict(
    :square     =>  [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    :honeycomb  =>  [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    :triangular =>  [[cosd(θ), sind(θ), 0.0] for θ in (90, 210, 330)],
    :kagome     =>  [[cosd(θ), sind(θ), 0.0] for θ in (90, 210, 330)],
)

default_kSize(lattice::Symbol) = lattice in (:square, :honeycomb) ? 33 : 27

"""A fresh spin-1/2 unit cell of the given lattice, without bonds."""
function new_cell(lattice::Symbol) :: UnitCell
    uc  =   UnitCell(GEOMETRY[lattice].primitives, 2, 2)
    for site in GEOMETRY[lattice].basis
        AddBasisSite!(uc, site)
    end
    return uc
end

"""The nearest-neighbour bonds of a unit cell, listed with AddIsotropicBonds! on a scratch Param."""
function nn_bonds(uc::UnitCell)
    nn  =   Param(1.0, 2)
    AddIsotropicBonds!(nn, uc, 1.0, SPIN[4], "nearest neighbours")
    return nn.unitBonds
end

bond_vector(uc::UnitCell, bond) = uc.basis[bond.target] - uc.basis[bond.base] + sum(bond.offset .* uc.primitives)


##### ------------------------------------------- Unrestricted order parameters -------------------------------------------

"""One complex Param per site and per channel."""
function onsite_orders(uc::UnitCell ; channels = HOPPING_CHANNELS) :: Vector{Param{2, ComplexF64}}
    orders  =   Param{2, ComplexF64}[]
    for site in eachindex(uc.basis), (name, mat) in channels
        channel =   Param(1.0 + 0.0im, 2)
        AddAnisotropicBond!(channel, uc, site, site, [0, 0], mat, 0.0, "$(name) on site $(site)")
        push!(orders, channel)
    end
    return orders
end

"""One complex Param per nearest-neighbour bond and per channel."""
function bond_orders(uc::UnitCell ; channels = BOND_CHANNELS) :: Vector{Param{2, ComplexF64}}
    orders  =   Param{2, ComplexF64}[]
    for bond in nn_bonds(uc), (name, mat) in channels
        channel =   Param(1.0 + 0.0im, 2)
        AddAnisotropicBond!(channel, uc, bond.base, bond.target, bond.offset, mat, 1.0, "$(name) on bond $(bond.base)→$(bond.target), $(bond.offset)")
        push!(orders, channel)
    end
    return orders
end


##### ------------------------------------------------------ Models ------------------------------------------------------

function add_nn_hopping!(uc::UnitCell, t::Float64)
    hopping =   Param(-t, 2)
    AddIsotropicBonds!(hopping, uc, 1.0, SPIN[4], "t1")
    CreateUnitCell!(uc, [hopping])
end

function nn_interaction(uc::UnitCell, strength::Float64, coupling, label::String)
    param   =   Param(strength, 4)
    AddIsotropicBonds!(param, uc, 1.0, coupling, label)
    return param
end

function filled_bz(uc::UnitCell, kSize::Int)
    bz  =   BZ([kSize, kSize])
    FillBZ!(bz, uc)
    return bz
end

"""Hubbard model -t Σ c†c + U Σ n↑n↓, with every on-site channel free (and bond channels tracked)."""
function hubbard_mft(lattice::Symbol ; t::Float64 = 1.0, U::Float64, filling::Float64 = 0.5, T::Float64 = 0.01, kSize::Int = default_kSize(lattice)) :: TBMFTModel
    uc      =   new_cell(lattice)
    add_nn_hopping!(uc, t)
    UParam  =   Param(U, 4)
    AddIsotropicBonds!(UParam, uc, 0.0, HUBBARD_U, "Hubbard Interaction")
    bz      =   filled_bz(uc, kSize)
    H       =   Hamiltonian(uc, bz)
    DiagonalizeHamiltonian!(H)
    model   =   Model(uc, bz, H ; T = T, filling = filling, stat = -1)
    return TBMFTModel(model, vcat(onsite_orders(uc), bond_orders(uc)), [UParam], IntraQuarticToHopping)
end

"""
Heisenberg model J Σ S_i·S_j with fermionic spinons at half filling, decoupled in every channel.
With `onsite = false` the on-site (magnetic) channels are scaled to zero: a spin-liquid ansatz.
"""
function heisenberg_mft(lattice::Symbol ; J::Float64 = 1.0, T::Float64 = 0.01, kSize::Int = default_kSize(lattice), onsite::Bool = true) :: TBMFTModel
    uc      =   new_cell(lattice)
    JParam  =   nn_interaction(uc, J, HEISENBERG_J, "Heisenberg Interaction")
    bz      =   filled_bz(uc, kSize)
    H       =   Hamiltonian(uc, bz)
    DiagonalizeHamiltonian!(H)
    model   =   Model(uc, bz, H ; T = T, filling = 0.5, stat = -1)
    scaling =   Dict{String, Float64}("ij" => 1.0, "ii" => onsite ? 1.0 : 0.0, "jj" => onsite ? 1.0 : 0.0)
    return TBMFTModel(model, vcat(onsite_orders(uc), bond_orders(uc)), [JParam], InterQuarticToHopping, scaling)
end

"""
Slave-boson t-J model at hole doping δ: spinons hop with δt at filling (1 - δ) per site, and J Σ S_i·S_j is decoupled
in every channel (magnetization, hopping, singlet and triplet pairing). `scheme = :gutzwiller` uses the renormalized
mean-field factors g_t = 2δ/(1+δ), g_J = 4/(1+δ)² instead. `pairing = false` switches the pairing channels off.
"""
function tJ_mft(lattice::Symbol ; t::Float64 = 1.0, J::Float64 = 0.3, δ::Float64, scheme::Symbol = :slaveboson, T::Float64 = 0.002,
                kSize::Int = lattice in (:square, :honeycomb) ? 27 : 21, pairing::Bool = true) :: BdGMFTModel
    t_eff, J_eff    =   scheme == :slaveboson ? (δ * t, J) : (2δ / (1 + δ) * t, 4J / (1 + δ)^2)
    hop, pair   =   new_cell(lattice), new_cell(lattice)
    add_nn_hopping!(hop, t_eff)
    JParam      =   nn_interaction(hop, J_eff, HEISENBERG_J, "Heisenberg Interaction")
    bz          =   filled_bz(hop, kSize)
    H           =   Hamiltonian(hop, pair, bz)
    DiagonalizeHamiltonian!(H)
    model       =   BdGModel(hop, pair, bz, H ; T = T, filling = (1 - δ) / 2, stat = -1)
    on, off     =   Dict("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), Dict("ij" => 0.0, "ii" => 0.0, "jj" => 0.0)
    return BdGMFTModel(model, vcat(onsite_orders(hop), bond_orders(hop)), bond_orders(pair ; channels = PAIRING_CHANNELS),
                       [JParam], Function[InterQuarticToHopping], Function[InterQuarticToPairing], on, pairing ? on : off)
end

order_labels(mft::TBMFTModel)   =   getproperty.(mft.HoppingOrders, :label)
order_labels(mft::BdGMFTModel)  =   vcat(getproperty.(mft.HoppingOrders, :label), getproperty.(mft.PairingOrders, :label))
norders(mft)                    =   length(order_labels(mft))
hopping_cell(mft::TBMFTModel)   =   mft.model.uc
hopping_cell(mft::BdGMFTModel)  =   mft.model.uc_hop

"""
Starting point for SolveMFT!: the lattice's classical spin pattern with moment `m` (channel value 2⟨S⟩), the density
from the filling, uniform bond hopping `χ`, and singlet pairing Δ e^{2iθ} on a bond at angle θ (d-wave on the square
lattice, d + id on the triangular and honeycomb lattices), so that no order is excluded by symmetry.
"""
function initial_guess(mft, lattice::Symbol ; m::Float64 = 0.9, χ::Float64 = 0.1, Δ::Float64 = 0.1) :: Vector{ComplexF64}
    uc      =   hopping_cell(mft)
    params  =   mft isa BdGMFTModel ? [mft.HoppingOrders ; mft.PairingOrders] : mft.HoppingOrders
    x       =   zeros(ComplexF64, length(params))
    for (i, param) in enumerate(params)
        name    =   first(split(param.label))
        bond    =   param.unitBonds[1]
        if name in ("Sx", "Sy", "Sz")
            x[i]    =   m * CLASSICAL_SPINS[lattice][bond.base][findfirst(==(name), ["Sx", "Sy", "Sz"])]
        elseif name == "n"
            x[i]    =   mft.model.filling
        elseif name == "χ"
            x[i]    =   χ
        elseif name == "Δs"
            r       =   bond_vector(uc, bond)
            x[i]    =   Δ * cis(2 * atan(r[2], r[1]))
        end
    end
    return x
end


##### ------------------------------------------------------ Analysis ------------------------------------------------------

"""On-site ⟨f†_a f_b⟩ for every site, rebuilt from the order parameters (exact: the on-site channels are a complete basis)."""
function onsite_correlations(mft) :: Vector{Matrix{ComplexF64}}
    lookup  =   Lookup(mft.HoppingOrders)
    return [ComplexF64.(get(lookup, (s, s, [0, 0]), zeros(ComplexF64, 2, 2))) for s in eachindex(hopping_cell(mft).basis)]
end

"""⟨S^μ⟩ = Σ_ab (S^μ)_ab ⟨f†_a f_b⟩ = tr((S^μ)ᵀ G) for G_ab = ⟨f†_a f_b⟩."""
spin_expectation(G::AbstractMatrix) = [real(tr(transpose(S) * G)) for S in SPIN[1:3]]

"""Per-site density n_i and spin vector ⟨S_i⟩."""
function site_moments(mft)
    Gs  =   onsite_correlations(mft)
    return (density = [real(tr(G)) for G in Gs], spin = spin_expectation.(Gs))
end

max_moment(mft) = maximum(norm.(site_moments(mft).spin))

"""A short description of the magnetic pattern from the site spins."""
function magnetic_pattern(spins ; threshold::Float64 = 1e-3) :: String
    moments =   norm.(spins)
    maximum(moments) < threshold && return "paramagnet"
    any(moments .< threshold) && return "partially ordered"
    units   =   normalize.(spins)
    cosines =   [dot(units[i], units[j]) for i in eachindex(units) for j in i+1:length(units)]
    total   =   norm(sum(spins)) / sum(moments)
    all(c -> c > 0.99, cosines)         && return "ferromagnet"
    all(c -> abs(c) > 0.99, cosines)    && return total < 0.05 ? "Néel" : "ferrimagnet"
    length(spins) == 3 && all(c -> isapprox(c, -0.5, atol = 0.02), cosines) && return "120° order"
    return "non-collinear"
end

"""Per nearest-neighbour bond: direction angle (degrees), singlet hopping χ, singlet pairing ψ and triplet size |d|."""
function bond_table(mft)
    uc      =   hopping_cell(mft)
    hop     =   Lookup(mft.HoppingOrders)
    pair    =   mft isa BdGMFTModel ? Lookup(mft.PairingOrders) : Dict()
    rows    =   NamedTuple[]
    for bond in nn_bonds(uc)
        key     =   (bond.base, bond.target, bond.offset)
        r       =   bond_vector(uc, bond)
        Δ       =   get(pair, key, zeros(ComplexF64, 2, 2))
        push!(rows, (bond = key, angle = rad2deg(atan(r[2], r[1])), χ = tr(get(hop, key, zeros(ComplexF64, 2, 2))) / 2,
                     ψ = tr(ISIGMAY' * Δ) / 2, triplet = norm([tr((2S * ISIGMAY)' * Δ) / 2 for S in SPIN[1:3]])))
    end
    return rows
end

pairing_amplitude(mft) = mft isa BdGMFTModel ? maximum(r -> abs(r.ψ), bond_table(mft)) : 0.0

"""Angular harmonic of the singlet pairing on the bonds: extended s (ℓ = 0), d-wave, or chiral d + id (ℓ = ±2)."""
function pairing_symmetry(rows ; threshold::Float64 = 1e-4) :: String
    ψs      =   getproperty.(rows, :ψ)
    maximum(abs, ψs ; init = 0.0) < threshold && return "no pairing"
    θs      =   deg2rad.(getproperty.(rows, :angle))
    θall    =   vcat(θs, θs .+ π)              ##### singlet pairing is even: ψ(θ + π) = ψ(θ)
    ψall    =   vcat(ψs, ψs)
    w(ℓ)    =   abs(sum(ψall .* cis.(-ℓ .* θall))) / sum(abs, ψall)
    w(0) > max(w(2), w(-2)) && return "extended s-wave"
    return min(w(2), w(-2)) < 0.1 * max(w(2), w(-2)) ? "d + id (chiral)" : "d-wave"
end
