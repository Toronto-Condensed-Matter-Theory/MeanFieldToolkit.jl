using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Distributions, Distributions, LaTeXStrings, Base.Threads
##### YOU NEED TO CREATE THE FOLDER Sample/Altermagnetism BEFORE RUNNING THIS SCRIPT
########## Defining square lattice with two orbitals
##### Primitives
const a1 = [1.0, 1.0]
const a2 = [1.0, -1.0]

const b1 = [0.0, 0.0]
const b2 = [1.0, 0.0]

SpinVec = SpinMats(1 // 2)

const t = 1.0
const t1 = 0.4 * t
const t2 = 0.2 * t

########## Brillouin Zone
const n = 20
const kSize = 6 * n + 3

##### Thermodynamic parameters
const T = 0.001
const filling = 0.5
const stat = -1

const mixingAlpha = 0.5

Us = collect(LinRange(1.0 * t, 6.0 * t, 41))

# Threads.@threads for U in Us
U = 4.0

HoppingUC = UnitCell([a1, a2], 2, 2)
ChiUC = UnitCell([a1, a2], 2, 2)
InteractionUC = UnitCell([a1, a2], 2, 4)

InitialField = zeros(Float64, HoppingUC.localDim^2)

AddBasisSite!(HoppingUC, b1)
AddBasisSite!(HoppingUC, b2)

AddBasisSite!(ChiUC, b1)
AddBasisSite!(ChiUC, b2)

AddBasisSite!(InteractionUC, b1)
AddBasisSite!(InteractionUC, b2)

######################## HoppingParams
tNNParam = Param(-t, 2)
AddIsotropicBonds!(tNNParam, HoppingUC, 1.0, SpinVec[4], "NN Hopping")

t1Param = Param(-t1, 2)
AddAnisotropicBond!(t1Param, HoppingUC, 1, 1, [1, 1], SpinVec[4], 2.0, "3NN Hopping t1")
AddAnisotropicBond!(t1Param, HoppingUC, 2, 2, [1, -1], SpinVec[4], 2.0, "3NN Hopping t1")

t2Param = Param(-t2, 2)
AddAnisotropicBond!(t2Param, HoppingUC, 1, 1, [1, -1], SpinVec[4], 2.0, "3NN Hopping t2")
AddAnisotropicBond!(t2Param, HoppingUC, 2, 2, [1, 1], SpinVec[4], 2.0, "3NN Hopping t2")

HoppingParams = [tNNParam, t1Param, t2Param]
CreateUnitCell!(HoppingUC, HoppingParams)

######################## Hopping expectation params
Neel = Param(1.0, 2)
AddAnisotropicBond!(Neel, ChiUC, 1, 1, [0, 0], SpinVec[3], 0.0, "Neel order")
AddAnisotropicBond!(Neel, ChiUC, 2, 2, [0, 0], -SpinVec[3], 0.0, "Neel order")

ChiParams = [Neel]

####################### Interaction Block                                  
n_up = [1.0 0.0; 0.0 0.0]
n_down = [0.0 0.0; 0.0 1.0]
Hubbard = DensityToPartonCoupling(n_up, n_down)

UParam = Param(U, 4)
AddIsotropicBonds!(UParam, InteractionUC, 0.0, Hubbard, "Hubbard Interaction")


bz = BZ([kSize, kSize])
FillBZ!(bz, HoppingUC)

H = Hamiltonian(HoppingUC, bz)
DiagonalizeHamiltonian!(H)

M = Model(HoppingUC, bz, H; T=T, filling=filling, stat=stat)
SolveModel!(M)

mft = TBMFTModel(M, ChiParams, [UParam], IntraQuarticToHopping)

fileName = "./Altermagnetism/U=$(round(U, digits=2))_t1=$(round(t1, digits=2))_t2=$(round(t2, digits=2))_new.jld2"
# SolveMFT!(mft, fileName)
SolveMFT!(mft)

# end
