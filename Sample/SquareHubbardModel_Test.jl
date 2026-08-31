using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Distributions, Base.Threads
using Plots
########## Defining square lattice 
##### Primitives
function doubled_uc()
    a1 = [1.0, 1.0]
    a2 = [1.0, -1.0]

    d1 = [0.0, 0.0]
    d2 = [1.0, 0.0]
    uc = UnitCell([a1, a2], 2, 2)
    for j = 0:1
        AddBasisSite!(uc,  j .* d2)
    end
    return uc
end

const InitialField = zeros(Float64, 4)

const n = 10
const kSize = 6 * n + 3

SpinVec = SpinMats(1 // 2)
const t = 1.0
const U = 2.0 *2

##### Thermodynamic parameters
const T = 0.00001
const stat = -1
const mixingAlpha = 0.5

UC =  doubled_uc()

fillings = collect(LinRange(0.1, 0.5, 20))

##### HoppingParams
t1 = -t
t1Param = Param(t1, 2)
AddIsotropicBonds!(t1Param, UC, 1.0, SpinVec[4], "t1")

HoppingParams = [t1Param]
n_up = [1.0 0.0; 0.0 0.0]
n_down = [0.0 0.0; 0.0 1.0]
Hubbard = DensityToPartonCoupling(n_up, n_down)

UParam = Param(U, 4)
AddIsotropicBonds!(UParam, UC, 0.0, Hubbard, "Hubbard Interaction")
CreateUnitCell!(UC, HoppingParams)

bz = BZ([kSize, kSize])
FillBZ!(bz, UC)
##### Hopping expectation params
t_s = Param(-1.0, 2)
AddIsotropicBonds!(t_s, UC, 1.0, SpinVec[4], "s Hopping")
# Neel = Param(1.0, 2)
# AddAnisotropicBond!(Neel, UC, 1, 1, [0, 0], SpinVec[3], 0.0, "Neel order")
# AddAnisotropicBond!(Neel, UC, 2, 2, [0, 0], -SpinVec[3], 0.0, "Neel order")

# ChiParams = [t_s, Neel]
Neel_1 = Param(1.0, 2)
Neel_2 = Param(1.0, 2)
AddAnisotropicBond!(Neel_1, UC, 1, 1, [0, 0], SpinVec[3], 0.0, "Neel order_1")
AddAnisotropicBond!(Neel_2, UC, 2, 2, [0, 0], SpinVec[3], 0.0, "Neel order_2")

ChiParams = [t_s, Neel_1, Neel_2]
####################### Interaction Block                                  
H = Hamiltonian(UC, bz)
DiagonalizeHamiltonian!(H)

M = Model(UC, bz, H; T=T, filling=0.5, stat=stat)
SolveModel!(M)
p = Plot_Band_Structure(M, [bz.HighSymPoints["G"], bz.HighSymPoints["M1"], bz.HighSymPoints["M2"]], labels=["G", "X", "M", "G"])
# display(p)
mft = TBMFTModel(M, ChiParams, [UParam], IntraQuarticToHopping)
# Define Initial OP
Init_OP_FM = [0.1,0.5,0.5]
Init_OP_AFM = [0.1,0.99,-0.99]

results = SolveMFT!(mft,Init_OP_AFM, max_iter = 1000, tol = 1E-6)#, fileName)
p = plot(mft.MFTEnergy, marker = ":circle",xlabel = "Iteration",ylabel = "MFT Energy",title = "Square Lattice Hubbard Model at U=4t, n=0.5")#, , , , legend = false)
display(p)
println("Sz on site 1: $(Neel_1.value[end])")
println("Sz on site 2: $(Neel_2.value[end])")
println("Total Energy as returned from SolveMFT function: $(mft.MFTEnergy[end])")
# Plot_Band_Structure(M, [bz.HighSymPoints["G"], bz.HighSymPoints["M1"], bz.HighSymPoints["M2"]], labels=["G", "X", "M", "G"])