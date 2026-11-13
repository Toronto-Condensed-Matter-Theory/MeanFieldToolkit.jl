##### Hubbard model on the honeycomb lattice at half filling: unrestricted Hartree-Fock.
##### Run from the repository root:   julia --project=Sample Sample/MinimalExamples/HoneycombHubbard.jl
using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging, Plots, LaTeXStrings
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log

########## Honeycomb lattice, with its two-site unit cell (Néel order fits)
const a1, a2    =   [sqrt(3) / 2, 1.5], [-sqrt(3) / 2, 1.5]
const sites     =   [[0.0, 0.0], [0.0, 1.0]]
const SpinVec   =   SpinMats(1 // 2)                         ##### [Sx, Sy, Sz, Id]

const t, U      =   1.0, 4.0
const T         =   0.01
const filling   =   0.5                                      ##### one electron per site
const kSize     =   33

UC  =   UnitCell([a1, a2], 2, 2)
for site in sites
    AddBasisSite!(UC, site)
end

##### Free nearest-neighbour hopping -t
t1  =   Param(-t, 2)
AddIsotropicBonds!(t1, UC, 1.0, SpinVec[4], "t1")
CreateUnitCell!(UC, [t1])

##### On-site Hubbard interaction U n↑ n↓
n_up, n_down    =   [1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]
UParam  =   Param(U, 4)
AddIsotropicBonds!(UParam, UC, 0.0, DensityToPartonCoupling(n_up, n_down), "Hubbard Interaction")

##### Order parameters: every spin channel on every site. One Param per site and per matrix in
##### [Sx, Sy, Sz, Id]; together they describe any on-site ⟨f†_a f_b⟩, so no magnetic pattern is assumed.
Channels    =   Param{2, Float64}[]
for site in eachindex(sites), (mat, name) in zip(SpinVec, ["Sx", "Sy", "Sz", "n"])
    channel =   Param(1.0, 2)
    AddAnisotropicBond!(channel, UC, site, site, [0, 0], mat, 0.0, "$(name) on site $(site)")
    push!(Channels, channel)
end

bz  =   BZ([kSize, kSize])
FillBZ!(bz, UC)
H   =   Hamiltonian(UC, bz)
DiagonalizeHamiltonian!(H)
M   =   Model(UC, bz, H ; T = T, filling = filling, stat = -1)

mft =   TBMFTModel(M, Channels, [UParam], IntraQuarticToHopping)

##### Start from a Néel state along z: value of the Sz channel = 2⟨Sz⟩, value of the n channel = n/2.
##### Order per site: [Sx, Sy, Sz, n]
Initial =   [0.0, 0.0, 0.5, 0.5,     0.0, 0.0, -0.5, 0.5]
sc      =   SolveMFT!(mft, Initial ; max_iter = 300, tol = 1e-8)
energies    =   copy(mft.MFTEnergy)                         ##### E/site after each iteration (the T loop below adds more)

println("U = $(U)t:  E/site = $(round(mft.MFTEnergy[end], digits = 6)) after $(length(sc.VOuts) - 1) iterations")
for (channel, value) in zip(Channels, sc.VOuts[end])
    println(rpad(channel.label, 14), round(value, digits = 5))
end

##### Finite temperature: raise T step by step, starting each temperature from the previous solution.
##### The Néel order m = 2|⟨S⟩| vanishes at the Néel temperature T_N.
Ts          =   range(0.05, 1.5, length = 20)
solutions   =   [sc.VOuts[end]]
moments     =   Float64[]
freeEnergy  =   Float64[]
for T in Ts
    mft.model.T =   T
    result      =   SolveMFT!(mft, solutions[end] ; max_iter = 300, tol = 1e-8)
    push!(solutions, result.VOuts[end])
    push!(moments, norm(result.VOuts[end][1:3]))           ##### |(Sx, Sy, Sz)| channels on site 1 = 2|⟨S_1⟩|
    push!(freeEnergy, GetMFTFreeEnergy(mft))                ##### F = E - TS per site
end
println("Néel temperature ≈ $(round(Ts[something(findfirst(<(1e-3), moments), length(Ts))], digits = 3))t (first T with m < 10⁻³)")

p0  =   plot(0:length(energies) - 1, energies, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}")
p1  =   plot(Ts, moments, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"m = 2|\langle \mathbf{S}_1 \rangle|")
p2  =   plot(Ts, freeEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"F = E - TS")
plot(p0, p1, p2, layout = (1, 3), size = (1300, 350), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
     plot_title = L"\mathrm{Honeycomb\ lattice\ Hubbard\ model},\ U = %$(U)t")
