using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Distributions, Distributions, LaTeXStrings, Base.Threads
##### YOU NEED TO CREATE THE FOLDER Sample/SquaretJ_Data BEFORE RUNNING THIS SCRIPT
################### Square lattice with the doubles Unit Cell
##### Primitives
const a1 = [1.0, 1.0]
const a2 = [1.0, -1.0]

const b1 = [0.0, 0.0]
const b2 = [1.0, 0.0]

HoppingUC = UnitCell([a1, a2], 2, 2)
PairingUC = UnitCell([a1, a2], 2, 2)

AddBasisSite!(HoppingUC, b1)
AddBasisSite!(HoppingUC, b2)

AddBasisSite!(PairingUC, b1)
AddBasisSite!(PairingUC, b2)

SpinVec = SpinMats(1 // 2)

##### HoppingParams
const t = 1.0
t1Param = Param(t, 2)
AddIsotropicBonds!(t1Param, HoppingUC, 1.0, SpinVec[4], "t1")

const J = t / 5
Jmatrix = Matrix{Float64}(I, 3, 3)
U = SpinToPartonCoupling(Jmatrix, 1 // 2)
JParam = Param(J, 4)
AddIsotropicBonds!(JParam, HoppingUC, 1.0, U, "Heisenberg Interaction")

const n = 10
const kSize = 6 * n + 3
bz = BZ([kSize, kSize])
FillBZ!(bz, HoppingUC)



##### Thermodynamic parameters
const T = 0.001
const stat = -1

const mixingAlpha = 0.5

##### Renormalized MFT from https://arxiv.org/pdf/cond-mat/0311604.pdf

fillings = collect(LinRange(0.2, 0.49, 2))

for filling in fillings

    delta = 1 - 2 * filling
    rnorm_hopping_factor = 2 * delta / (1 + delta)
    rnorm_int_factor = 4 / ((1 + delta)^2)

    push!(t1Param.value, -t * rnorm_hopping_factor)
    push!(JParam.value, J * rnorm_int_factor)

    ModifyUnitCell!(HoppingUC, [t1Param])
    Interactions = [JParam]

    ##### Hopping expectation params
    t_s = Param(1.0, 2)
    AddIsotropicBonds!(t_s, HoppingUC, 1.0, SpinVec[4], "s Hopping")

    HoppingOrders = [t_s]

    ##### Pairing expectation params
    p_dx2y2 = Param(1.0, 2)
    AddAnisotropicBond!(p_dx2y2, PairingUC, 1, 2, [0, 0], (SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, PairingUC, 1, 2, [-1, -1], (SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, PairingUC, 1, 2, [0, -1], -(SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")
    AddAnisotropicBond!(p_dx2y2, PairingUC, 1, 2, [-1, 0], -(SpinVec[2]), 1.0, "d_x^2-y^2 Pairing")

    PairingOrders = [p_dx2y2]

    bdgH = Hamiltonian(HoppingUC, PairingUC, bz)
    DiagonalizeHamiltonian!(bdgH)

    bdgModel = BdGModel(HoppingUC, PairingUC, bz, bdgH; T=T, filling=filling, stat=stat)
    SolveModel!(bdgModel)

    bdgmft = BdGMFTModel(bdgModel, HoppingOrders, PairingOrders, Interactions, InterQuarticToHopping, InterQuarticToPairing)
    fileName = "./SquaretJ_Data/filling=$(round(filling, digits=3))_t1=$(round(t1Param.value[end], digits=3))_J=$(round(J, digits=3))_wtWeiss.jld2"
    SolveMFT!(bdgmft, fileName)

end