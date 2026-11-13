##### Heisenberg model on the triangular lattice with fermionic spinons S = ½ f† σ f: unrestricted mean-field theory.
##### Run from the repository root:   julia --project=Sample Sample/MinimalExamples/TriangularHeisenberg.jl
using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging, Plots, LaTeXStrings
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log

########## Triangular lattice, with a √3 × √3 unit cell (three sites) so that 120° order fits
const a1, a2    =   [1.5, sqrt(3) / 2], [0.0, sqrt(3)]
const sites     =   [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2]]
const SpinVec   =   SpinMats(1 // 2)                         ##### [Sx, Sy, Sz, Id]

const J         =   1.0
const T         =   0.01
const kSize     =   27

UC  =   UnitCell([a1, a2], 2, 2)
for site in sites
    AddBasisSite!(UC, site)
end

##### Nearest-neighbour exchange J S_i·S_j, written as a quartic spinon interaction (no free hopping)
JParam  =   Param(J, 4)
AddIsotropicBonds!(JParam, UC, 1.0, SpinToPartonCoupling(Matrix{Float64}(I, 3, 3), 1 // 2), "Heisenberg Interaction")

##### Order parameters, one complex Param per matrix in [Sx, Sy, Sz, Id]:
#####   on every site (magnetization, density), and on every nearest-neighbour bond (spinon hopping χ, singlet and triplet).
Channels    =   Param{2, ComplexF64}[]
for site in eachindex(sites), (mat, name) in zip(SpinVec, ["Sx", "Sy", "Sz", "n"])
    channel =   Param(1.0 + 0.0im, 2)
    AddAnisotropicBond!(channel, UC, site, site, [0, 0], mat, 0.0, "$(name) on site $(site)")
    push!(Channels, channel)
end
nn  =   Param(1.0, 2)                                        ##### a scratch Param, just to list the nearest-neighbour bonds
AddIsotropicBonds!(nn, UC, 1.0, SpinVec[4], "nearest neighbours")
for bond in nn.unitBonds, (mat, name) in zip(SpinVec, ["χx", "χy", "χz", "χ"])
    channel =   Param(1.0 + 0.0im, 2)
    AddAnisotropicBond!(channel, UC, bond.base, bond.target, bond.offset, mat, 1.0, "$(name) on bond $(bond.base)→$(bond.target), $(bond.offset)")
    push!(Channels, channel)
end

bz  =   BZ([kSize, kSize])
FillBZ!(bz, UC)
H   =   Hamiltonian(UC, bz)
DiagonalizeHamiltonian!(H)
M   =   Model(UC, bz, H ; T = T, filling = 0.5, stat = -1)   ##### one spinon per site (on average)

##### Two decouplings of the same model: every channel (magnetic order allowed),
##### and spinon hopping only (on-site channels scaled to zero: a spin-liquid ansatz). From the uniform start below the
##### latter finds the uniform RVB state; other starting points find dimerized states, which are lower in mean-field theory.
magnetic    =   TBMFTModel(M, Channels, [JParam], InterQuarticToHopping)
liquid      =   TBMFTModel(deepcopy(M), deepcopy(Channels), [JParam], InterQuarticToHopping, Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0))

##### Start from the 120° state in the xy plane (S channels = 2⟨S⟩) with a weak uniform spinon hopping
Initial =   zeros(ComplexF64, length(Channels))
for (i, channel) in enumerate(Channels)
    startswith(channel.label, "n ") && (Initial[i] = 0.5)
    startswith(channel.label, "χ ") && (Initial[i] = 0.1)
end
for (site, θ) in enumerate([90, 210, 330])                     ##### Sx, Sy on each site
    Initial[4site-3], Initial[4site-2]  =   0.9cosd(θ), 0.9sind(θ)
end

scMagnetic  =   SolveMFT!(magnetic, Initial ; max_iter = 300, tol = 1e-8)
energies    =   copy(magnetic.MFTEnergy)                         ##### E/site after each iteration (the T loop below adds more)
scLiquid    =   SolveMFT!(liquid, Initial ; max_iter = 300, tol = 1e-8)
spins   =   [norm(scMagnetic.VOuts[end][4site-3:4site-1]) / 2 for site in eachindex(sites)]           ##### |⟨S⟩| per site
χs      =   abs.(scLiquid.VOuts[end][4length(sites)+4:4:end])                                       ##### |χ| per bond
println("All channels:        E/site = $(round(magnetic.MFTEnergy[end], digits = 6)),  |⟨S⟩| per site = $(round.(spins, digits = 4))")
println("Spinon hopping only: E/site = $(round(liquid.MFTEnergy[end], digits = 6)),  |χ| per bond = $(round.(χs, digits = 4))")

##### Finite temperature: the 120° order melts at T_N = z|cos θ|J/8 = 0.375J for fermionic spinons (z = 6 neighbours
##### at θ = 120°; half the spin-1/2 Weiss value, because one spinon per site holds only on average).
Ts          =   range(0.05, 0.55, length = 11)
solutions   =   [scMagnetic.VOuts[end]]
moments     =   Float64[]
for T in Ts
    magnetic.model.T    =   T
    result  =   SolveMFT!(magnetic, solutions[end] ; max_iter = 300, tol = 1e-8)
    push!(solutions, result.VOuts[end])
    push!(moments, norm(result.VOuts[end][1:3]) / 2)        ##### |⟨S_1⟩|
end
println("Ordering temperature ≈ $(round(Ts[something(findfirst(<(1e-3), moments), length(Ts))], digits = 3))J (first T with |⟨S⟩| < 10⁻³; mean-field value 0.375J)")

p0  =   plot(0:length(energies) - 1, energies, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}")
p1  =   plot(Ts, moments, marker = :circle, lw = 2, legend = false, xlabel = L"T/J", ylabel = L"|\langle \mathbf{S}_1 \rangle|")
plot(p0, p1, layout = (1, 2), size = (900, 350), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, plot_title = L"\mathrm{Triangular\ lattice\ Heisenberg\ model\ (fermionic\ spinons)}")
