include("../src/MeanFieldToolkit.jl")
using .MeanFieldToolkit
using TightBindingToolkit, FixedPointToolkit, JLD2, Plots, LaTeXStrings

##### primitive vectors
const a1  =   [1/2, sqrt(3)/2]
const a2  =   [-1/2, sqrt(3)/2]

const SpinVec = SpinMats(1//2)
##### sublattices
const b1  =   [0.0, 0.0]
const b2  =   [0.0, 1/sqrt(3)]

function honeycomb(t1::Float64, t3::Float64, inPlaneField::Float64, outPlaneField::Float64)
    HoppingUC       =   UnitCell([a1, a2], 2, 2)
    AddBasisSite!.(Ref(HoppingUC), [b1, b2])

    tNN = Param(t1, 2)
    t3NN = Param(t3, 2)
    InPlaneField = Param(inPlaneField, 2)
    OutPlaneField = Param(outPlaneField, 2)

    AddIsotropicBonds!(tNN, HoppingUC, 1/sqrt(3), 2*SpinVec[3], "NN Hopping")
    AddIsotropicBonds!(t3NN, HoppingUC, 2/sqrt(3), 2*SpinVec[3], "3NN Hopping")
    AddIsotropicBonds!(InPlaneField, HoppingUC, 0.0, SpinVec[1], "InPlane Field")
    AddIsotropicBonds!(OutPlaneField, HoppingUC, 0.0, SpinVec[3], "OutPlane Field")

    CreateUnitCell!(HoppingUC,[tNN,t3NN,InPlaneField,OutPlaneField])

    return HoppingUC
end

##### Thermodynamic parameters
const T         =   0.05
const stat      =   -1
const filling   = 0.5

function honeycombMFT(t1::Float64, t3::Float64, inPlaneField::Float64, outPlaneField::Float64,
                        J1::Float64, J3::Float64,
                        fileName::String,
                        scalings::Dict{String, Any} = Dict{String, Any}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0))
    HoppingUC = honeycomb(t1, t3, inPlaneField, outPlaneField)
    bz = BZ([33, 33])
    FillBZ!(bz, HoppingUC)

    Jmatrix     =   [[1.0 0.0 0.0];[0.0 1.0 0.0];[0.0 0.0 0.0]]
    U           =   SpinToPartonCoupling(Jmatrix, 1//2)
    JParam3      =   Param(J3, 4)
    JParam1      =   Param(J1, 4)
    AddIsotropicBonds!(JParam3,HoppingUC, 2/sqrt(3), U, "J3 Interaction")
    AddIsotropicBonds!(JParam1,HoppingUC, 1/sqrt(3), U, "J1 Interaction")
    InteractionParams   =  [JParam3,JParam1]

    ##### Order parameters
    ferro = Param(1.0, 2)
    direction = 1
    ##### Ferromagnetic order
    ferroSigns = [1, 1]
    for (b, basis) in enumerate(HoppingUC.basis)
        AddAnisotropicBond!(ferro, HoppingUC, b, b, [0, 0], ferroSigns[b] * SpinVec[direction], 0.0, "ferro order along $(direction)")
    end

    t1Chi = Param(1.0, 2)
    AddIsotropicBonds!(t1Chi, HoppingUC, 1/sqrt(3), 2*SpinVec[3], "t1 expectation")
    t3Chi = Param(1.0, 2)
    AddIsotropicBonds!(t3Chi, HoppingUC, 2/sqrt(3), 2*SpinVec[3], "t3 expectation")

    expectations = [ferro, t1Chi, t3Chi]

    H           =   Hamiltonian(HoppingUC, bz)
    DiagonalizeHamiltonian!(H)

    model    =   Model(HoppingUC, bz, H ; T=T, filling=filling, stat=stat)
    mft      =   TBMFTModel(model, expectations, InteractionParams, InterQuarticToHopping, scalings)

    if fileName != ""
        sc = SolveMFT!(mft, fileName; max_iter=200, tol=1e-4);
    else
        sc = SolveMFT!(mft; max_iter=200, tol=1e-4)
    end

    return mft

end

input = load("./J1=-1.0_J3=0.3_T=0.05_wBx_Scaling=0.0.jld2")


##################### parameters
const J = 1.0
const g = 1.0
const alpha1 = 0.75
const alpha3 = 0.7*alpha1

const t1 = 0.0
const t3 = 0.0
const outPlaneField = 0.0

const J1 = -1.0*J*g
const J3 = 0.3*J*g

const inPlanes = collect(range(0.0, 0.5, length=51))*J

ferros = Float64[]
t1s = Float64[]
t3s = Float64[]
energies = Float64[]

scalings1 = Dict{String, Float64}("ij" => alpha1, "ii" => 1.0 - alpha1, "jj" => 1.0 - alpha1)
scalings3 = Dict{String, Float64}("ij" => alpha3, "ii" => 1.0 - alpha3, "jj" => 1.0 - alpha3)
scalings = Dict{String, Any}("J1 Interaction" => scalings1, "J3 Interaction" => scalings3)



for (b, Bx) in enumerate(inPlanes)
    fileName = ""
    sc = honeycombMFT(0.0, 0.0, Bx, outPlaneField, J1, J3, fileName, scalings)
    push!(ferros, sc.HoppingOrders[1].value[end])
    push!(t1s, sc.HoppingOrders[2].value[end])
    push!(t3s, sc.HoppingOrders[3].value[end])
    push!(energies, sc.MFTEnergy[end])
    println("Bx = $(Bx) done")
end
