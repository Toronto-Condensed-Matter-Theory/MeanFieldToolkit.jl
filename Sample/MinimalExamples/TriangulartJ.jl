##### t-J model on the triangular lattice in slave-boson mean-field theory: c_iσ = b_i† f_iσ with condensed bosons ⟨b⟩² = δ,
##### so the spinons hop with δt at filling (1 - δ) per site, and J S_i·S_j is decoupled in every spinon channel
##### (magnetization, hopping χ, singlet and triplet pairing Δ).
##### Run from the repository root:   julia --project=Sample Sample/MinimalExamples/TriangulartJ.jl
using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging, Plots, LaTeXStrings
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log

########## Triangular lattice, with a √3 × √3 unit cell (three sites) so that 120° order fits
const a1, a2    =   [1.5, sqrt(3) / 2], [0.0, sqrt(3)]
const sites     =   [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2]]
const SpinVec   =   SpinMats(1 // 2)                         ##### [Sx, Sy, Sz, Id]
const iσy       =   ComplexF64[0 1; -1 0]                    ##### spin-singlet pairing; 2 S^μ iσy are the triplets

const t, J, δ   =   1.0, 0.3, 0.05
const T         =   0.002
const kSize     =   21

HoppingUC   =   UnitCell([a1, a2], 2, 2)
PairingUC   =   UnitCell([a1, a2], 2, 2)
for site in sites
    AddBasisSite!(HoppingUC, site)
    AddBasisSite!(PairingUC, site)
end

##### Spinon hopping δt, and the exchange J S_i·S_j as a quartic spinon interaction
t1  =   Param(-δ * t, 2)
AddIsotropicBonds!(t1, HoppingUC, 1.0, SpinVec[4], "t1")
CreateUnitCell!(HoppingUC, [t1])
JParam  =   Param(J, 4)
AddIsotropicBonds!(JParam, HoppingUC, 1.0, SpinToPartonCoupling(Matrix{Float64}(I, 3, 3), 1 // 2), "Heisenberg Interaction")

##### Order parameters: every channel on every site and nearest-neighbour bond (hopping),
##### and the singlet and three triplet pairing channels on every bond.
HoppingOrders   =   Param{2, ComplexF64}[]
PairingOrders   =   Param{2, ComplexF64}[]
for site in eachindex(sites), (mat, name) in zip(SpinVec, ["Sx", "Sy", "Sz", "n"])
    channel =   Param(1.0 + 0.0im, 2)
    AddAnisotropicBond!(channel, HoppingUC, site, site, [0, 0], mat, 0.0, "$(name) on site $(site)")
    push!(HoppingOrders, channel)
end
nn  =   Param(1.0, 2)                                        ##### a scratch Param, just to list the nearest-neighbour bonds
AddIsotropicBonds!(nn, HoppingUC, 1.0, SpinVec[4], "nearest neighbours")
for bond in nn.unitBonds
    for (mat, name) in zip(SpinVec, ["χx", "χy", "χz", "χ"])
        channel =   Param(1.0 + 0.0im, 2)
        AddAnisotropicBond!(channel, HoppingUC, bond.base, bond.target, bond.offset, mat, 1.0, "$(name) on bond $(bond.base)→$(bond.target), $(bond.offset)")
        push!(HoppingOrders, channel)
    end
    for (mat, name) in zip([iσy, 2SpinVec[1] * iσy, 2SpinVec[2] * iσy, 2SpinVec[3] * iσy], ["Δs", "Δx", "Δy", "Δz"])
        channel =   Param(1.0 + 0.0im, 2)
        AddAnisotropicBond!(channel, PairingUC, bond.base, bond.target, bond.offset, mat, 1.0, "$(name) on bond $(bond.base)→$(bond.target), $(bond.offset)")
        push!(PairingOrders, channel)
    end
end

bz  =   BZ([kSize, kSize])
FillBZ!(bz, HoppingUC)
H   =   Hamiltonian(HoppingUC, PairingUC, bz)
DiagonalizeHamiltonian!(H)
M   =   BdGModel(HoppingUC, PairingUC, bz, H ; T = T, filling = (1 - δ) / 2, stat = -1)
mft =   BdGMFTModel(M, HoppingOrders, PairingOrders, [JParam], InterQuarticToHopping, InterQuarticToPairing)

##### Start with weak magnetic order, uniform hopping, and singlet pairing Δ ∝ e^{2iθ} on a bond at angle θ
##### (d-wave for the square lattice, d + id for the triangular and honeycomb lattices), so no order is excluded by symmetry.
Initial =   zeros(ComplexF64, length(HoppingOrders) + length(PairingOrders))
for (i, channel) in enumerate([HoppingOrders ; PairingOrders])
    startswith(channel.label, "n ") && (Initial[i] = (1 - δ) / 2)
    startswith(channel.label, "χ ") && (Initial[i] = 0.1)
    if startswith(channel.label, "Δs ")
        bond        =   channel.unitBonds[1]
        r           =   HoppingUC.basis[bond.target] - HoppingUC.basis[bond.base] + sum(bond.offset .* HoppingUC.primitives)
        Initial[i]  =   0.1 * cis(2 * atan(r[2], r[1]))
    end
end
for (site, θ) in enumerate([90, 210, 330])                     ##### weak 120° order: Sx, Sy on each site
    Initial[4site-3], Initial[4site-2]  =   0.1cosd(θ), 0.1sind(θ)
end

sc  =   SolveMFT!(mft, Initial ; max_iter = 500, tol = 1e-8)
energies    =   copy(mft.MFTEnergy)                         ##### E/site after each iteration (the T loop below adds more)
println("δ = $(δ):  E/site = $(round(mft.MFTEnergy[end], digits = 6)) after $(length(sc.VOuts) - 1) iterations")
for (channel, value) in zip(PairingOrders, sc.VOuts[end][length(HoppingOrders)+1:end])
    startswith(channel.label, "Δs ") && println(rpad(channel.label, 30), round(value, digits = 4))
end

##### Finite temperature: the singlet pairing vanishes at the spinon pairing temperature T_pair
Ts          =   range(0.002, 0.05, length = 20)
solutions   =   [sc.VOuts[end]]
pairing     =   Float64[]
for T in Ts
    mft.model.T =   T
    result  =   SolveMFT!(mft, solutions[end] ; max_iter = 500, tol = 1e-8)
    push!(solutions, result.VOuts[end])
    push!(pairing, maximum(abs, result.VOuts[end][length(HoppingOrders)+1:4:end]))    ##### the Δs channels
end
println("T_pair ≈ $(round(Ts[something(findfirst(<(1e-4), pairing), length(Ts))], digits = 4))t (first T with |Δs| < 10⁻⁴)")

p0  =   plot(0:length(energies) - 1, energies, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}")
p1  =   plot(Ts, pairing, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\max |\Delta_s|")
plot(p0, p1, layout = (1, 2), size = (900, 350), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, plot_title = L"\mathrm{Triangular\ lattice}\ t\textrm{-}J,\ \delta = %$(δ),\ J = %$(J)t")
