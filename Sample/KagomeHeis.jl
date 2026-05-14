using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Distributions, Distributions, LaTeXStrings, Base.Threads
##### YOU NEED TO CREATE THE FOLDER Sample/KagomeHeis_Data BEFORE RUNNING THIS SCRIPT
########## Defining Kagome lattice with the doubled unit cell
##### Primitives
const a1 = [4.0, 0.0]
const a2 = [1.0, sqrt(3)]
##### 6 sublattices
const b1 = [0.0, 0.0]
const b2 = [1.0, 0.0]
const b3 = [0.5, sqrt(3) / 2]
const b4 = [2.0 + 0.0, 0.0]
const b5 = [2.0 + 1.0, 0.0]
const b6 = [2.0 + 0.5, sqrt(3) / 2]
##### On-site spin matrices
const InitialField = zeros(Float64, 4)
const OnSite = SpinMats(1 // 2)
##### Unit cell with local dimensions of 2, and tracking rank-2 bonds
UC = UnitCell([a1, a2], 2, 2)
AddBasisSite!.(Ref(UC), [b1, b2, b3, b4, b5, b6], Ref(InitialField), Ref(OnSite))
##### Strength of the Heisenberg interaction
const J = 1.0
##### Heisenberg spin exchange matrix
Jmatrix = Matrix{Float64}(I, 3, 3)
##### Heisenberg interaction written in terms of 4-parton interactions --> rank-4 array.
U = SpinToPartonCoupling(Jmatrix, 1 // 2)
###### Interaction parameter which is isotropic.                               
JParam = Param(J, 4)
AddIsotropicBonds!(JParam, UC, 1.0, U, "Heisenberg Interaction")
Interactions = [JParam]
##### Brillouin zone
const n = 10
const kSize = 6 * n + 3
bz = BZ([kSize, kSize])
FillBZ!(bz, UC)
##### Thermodynamic parameters
const T = 0.001
const filling = 0.5
const stat = -1
##### Mixing alpha used for self-consistency solver
const mixingAlpha = 0.5
##### Different flux configuration ansatzes to be tested.
phis = collect(LinRange(-pi / 2, pi / 2, 21))

for ϕ in phis

    ##### Hopping expectation params
    ##### Hopping with flux ϕ through the triangles, and π-2ϕ through the hexagons
    t_flux = Param(1.0, 2)
    AddAnisotropicBond!(t_flux, UC, 1, 2, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 2, 3, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 3, 1, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 4, 5, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 5, 6, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 6, 4, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 4, 2, [0, 0], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")

    AddAnisotropicBond!(t_flux, UC, 2, 6, [0, -1], exp(im * (pi + ϕ / 3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 4, 6, [0, -1], exp(-im * (pi + ϕ / 3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 5, 3, [1, -1], exp(im * (pi + ϕ / 3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 5, 1, [1, 0], exp(-im * (pi + ϕ / 3)) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")
    AddAnisotropicBond!(t_flux, UC, 3, 1, [0, 1], exp(im * ϕ / 3) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2*ϕ] Hopping with ϕ/π = $(ϕ/pi)")

    ######### Weiss fields for ordering
    Ordering = Param(1.0, 2)
    AddAnisotropicBond!(Ordering, UC, 1, 1, [0, 0], sum([-sqrt(3) / 2, -1 / 2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, UC, 2, 2, [0, 0], sum([sqrt(3) / 2, -1 / 2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, UC, 3, 3, [0, 0], sum([0.0, 1.0, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, UC, 4, 4, [0, 0], sum([-sqrt(3) / 2, -1 / 2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, UC, 5, 5, [0, 0], sum([sqrt(3) / 2, -1 / 2, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")
    AddAnisotropicBond!(Ordering, UC, 6, 6, [0, 0], sum([0.0, 1.0, 0.0] .* OnSite[1:3]), 0.0, "120 degree ordering")

    HoppingOrders = [t_flux, Ordering]
    ##### Build the model (trivial since we are looking at the pure spin  Heisenberg model)
    H = Hamiltonian(UC, bz)
    DiagonalizeHamiltonian!(H)
    M = Model(UC, bz, H; T=T, filling=filling, stat=stat)
    ##### Build the mean-field theory object, with chosen scaling such that on-site ordering is suppressed.
    mft = TBMFTModel(M, HoppingOrders, Interactions, Function[InterQuarticToHopping], Dict{String,Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0))
    ##### File to save data to
    fileName = "./KagomeHeis_Data/J=$(round(J, digits=3))_phi=$(round(ϕ/pi, digits=3))Pi_New.jld2"
    ##### Solve the mean-field theory and save the results in fileName
    SolveMFT!(mft, fileName)

end
