##### Altermagnetism from a Hubbard model on the square lattice. Two sublattices with nearest-neighbour hopping t, plus
##### third-neighbour hoppings that differ between the sublattices (t1 along one diagonal, t2 along the other). A Néel state
##### then splits the bands by spin in a d-wave pattern with zero net magnetization: an altermagnet.
##### Only the Néel channel is decoupled, so the energy omits the constant Hartree term U n²/4.
##### Run from the repository root:   julia --project=Sample Sample/MinimalExamples/SquareAltermagnet.jl
using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging, Plots, LaTeXStrings
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log

########## Square lattice with two orbitals (sublattices)
const a1, a2    =   [1.0, 1.0], [1.0, -1.0]
const b1, b2    =   [0.0, 0.0], [1.0, 0.0]
const SpinVec   =   SpinMats(1 // 2)

const t, t1, t2 =   1.0, 0.4, 0.2
const U         =   6.0
const T         =   0.001
const filling   =   0.5
const kSize     =   63

HoppingUC       =   UnitCell([a1, a2], 2, 2)
ChiUC           =   UnitCell([a1, a2], 2, 2)
InteractionUC   =   UnitCell([a1, a2], 2, 4)
for uc in (HoppingUC, ChiUC, InteractionUC)
    AddBasisSite!(uc, b1)
    AddBasisSite!(uc, b2)
end

######################## Hopping params: NN, and sublattice-dependent third neighbours
tNNParam    =   Param(-t, 2)
AddIsotropicBonds!(tNNParam, HoppingUC, 1.0, SpinVec[4], "NN Hopping")
t1Param     =   Param(-t1, 2)
AddAnisotropicBond!(t1Param, HoppingUC, 1, 1, [1, 1],  SpinVec[4], 2.0, "3NN Hopping t1")
AddAnisotropicBond!(t1Param, HoppingUC, 2, 2, [1, -1], SpinVec[4], 2.0, "3NN Hopping t1")
t2Param     =   Param(-t2, 2)
AddAnisotropicBond!(t2Param, HoppingUC, 1, 1, [1, -1], SpinVec[4], 2.0, "3NN Hopping t2")
AddAnisotropicBond!(t2Param, HoppingUC, 2, 2, [1, 1],  SpinVec[4], 2.0, "3NN Hopping t2")
CreateUnitCell!(HoppingUC, [tNNParam, t1Param, t2Param])

######################## Néel order parameter
Neel    =   Param(1.0, 2)
AddAnisotropicBond!(Neel, ChiUC, 1, 1, [0, 0],  SpinVec[3], 0.0, "Neel order")
AddAnisotropicBond!(Neel, ChiUC, 2, 2, [0, 0], -SpinVec[3], 0.0, "Neel order")

####################### On-site Hubbard interaction
n_up, n_down    =   [1.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 1.0]
UParam  =   Param(U, 4)
AddIsotropicBonds!(UParam, InteractionUC, 0.0, DensityToPartonCoupling(n_up, n_down), "Hubbard Interaction")

bz  =   BZ([kSize, kSize])
FillBZ!(bz, HoppingUC)
H   =   Hamiltonian(HoppingUC, bz)
DiagonalizeHamiltonian!(H)
M   =   Model(HoppingUC, bz, H ; T = T, filling = filling, stat = -1)

mft =   TBMFTModel(M, [Neel], [UParam], IntraQuarticToHopping)
sc  =   SolveMFT!(mft, [0.5] ; max_iter = 300, tol = 1e-8)
energies    =   copy(mft.MFTEnergy)                         ##### E/site after each iteration (the T loop below adds more)

##### Spin splitting E↓ - E↑ of the lowest pair of bands at X = (π/2, 0) and Y = (0, π/2):
##### zero in a conventional antiferromagnet, equal and opposite at X and Y in an altermagnet.
Sz  =   kron(Matrix{Float64}(I, 2, 2), 2 .* SpinVec[3])
for (name, k) in (("X", [π / 2, 0.0]), ("Y", [0.0, π / 2]))
    E, V    =   eigen(Hermitian(FillHoppingHamiltonian(mft.model.uc, k)))
    sz      =   [real(V[:, n]' * Sz * V[:, n]) for n in 1:2]
    println("Spin splitting at $(name): $(round((E[2] - E[1]) * sign(sz[1] - sz[2]), digits = 4))t")
end
println("U = $(U)t:  Néel order m = $(round(sc.VOuts[end][1], digits = 5)),  E/site = $(round(mft.MFTEnergy[end], digits = 6))")

##### Finite temperature: raise T step by step, starting each temperature from the previous solution
Ts          =   range(0.05, 1.5, length = 20)
solutions   =   [sc.VOuts[end]]
for T in Ts
    mft.model.T =   T
    result      =   SolveMFT!(mft, solutions[end] ; max_iter = 300, tol = 1e-8)
    push!(solutions, result.VOuts[end])
end
moments =   abs.(first.(solutions[2:end]))
println("Néel temperature ≈ $(round(Ts[something(findfirst(<(1e-3), moments), length(Ts))], digits = 3))t (first T with m < 10⁻³)")

p0  =   plot(0:length(energies) - 1, energies, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}")
p1  =   plot(Ts, moments, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\mathrm{N\acute{e}el\ order}\ m")
plot(p0, p1, layout = (1, 2), size = (900, 350), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, plot_title = L"\mathrm{Square\ lattice\ altermagnet},\ U = %$(U)t")
