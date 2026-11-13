##### Kagome Heisenberg model: free energy of fermionic-spinon flux states.
##### The spinon hopping has a fixed pattern (an ansatz) whose amplitude is found self-consistently: flux ϕ through each
##### triangle and π - 2ϕ through each hexagon, on a doubled (6-site) unit cell. ϕ = 0 is the U(1) Dirac spin liquid and
##### ϕ = π/2 the chiral spin liquid. On-site (magnetic) channels are scaled to zero, so these are spin-liquid states.
##### Run from the repository root:   julia --project=Sample Sample/MinimalExamples/KagomeFluxScan.jl
using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
using LinearAlgebra, Logging, Plots, LaTeXStrings
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log

########## Kagome lattice with a doubled unit cell (six sites)
const a1, a2    =   [4.0, 0.0], [1.0, sqrt(3)]
const sites     =   [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2], [2.0, 0.0], [3.0, 0.0], [2.5, sqrt(3) / 2]]
const OnSite    =   SpinMats(1 // 2)

const J         =   1.0
const kSize     =   33
const ϕs        =   range(0.0, π / 2, length = 7)
const Ts        =   [0.001, 0.05, 0.1, 0.15]

FreeEnergy  =   zeros(length(ϕs), length(Ts))
energies    =   Float64[]                                   ##### E/site per iteration, for ϕ = π/2 at the lowest T
for (i, ϕ) in enumerate(ϕs)
    UC  =   UnitCell([a1, a2], 2, 2)
    for site in sites
        AddBasisSite!(UC, site)
    end

    ##### Heisenberg interaction J S_i·S_j as a quartic spinon interaction
    JParam  =   Param(J, 4)
    AddIsotropicBonds!(JParam, UC, 1.0, SpinToPartonCoupling(Matrix{Float64}(I, 3, 3), 1 // 2), "Heisenberg Interaction")

    ##### Hopping with flux ϕ through the triangles, and π - 2ϕ through the hexagons
    t_flux  =   Param(1.0, 2)
    for (base, target, offset, phase) in ((1, 2, [0, 0], ϕ / 3), (2, 3, [0, 0], ϕ / 3), (3, 1, [0, 0], ϕ / 3),
                                          (4, 5, [0, 0], ϕ / 3), (5, 6, [0, 0], ϕ / 3), (6, 4, [0, 0], ϕ / 3), (4, 2, [0, 0], ϕ / 3),
                                          (2, 6, [0, -1], π + ϕ / 3), (4, 6, [0, -1], -(π + ϕ / 3)), (5, 3, [1, -1], π + ϕ / 3),
                                          (5, 1, [1, 0], -(π + ϕ / 3)), (3, 1, [0, 1], ϕ / 3))
        AddAnisotropicBond!(t_flux, UC, base, target, offset, cis(phase) * OnSite[4], 1.0, "[ϕ, ϕ, π - 2ϕ] Hopping")
    end

    ##### Weiss field for 120° order, tracked for reference (on-site channels are scaled to zero below)
    Ordering    =   Param(1.0, 2)
    directions  =   ([-sqrt(3) / 2, -1 / 2, 0.0], [sqrt(3) / 2, -1 / 2, 0.0], [0.0, 1.0, 0.0])
    for site in eachindex(sites)
        AddAnisotropicBond!(Ordering, UC, site, site, [0, 0], sum(directions[mod1(site, 3)] .* OnSite[1:3]), 0.0, "120 degree ordering")
    end

    bz  =   BZ([kSize, kSize])
    FillBZ!(bz, UC)
    H   =   Hamiltonian(UC, bz)
    DiagonalizeHamiltonian!(H)
    M   =   Model(UC, bz, H ; T = Ts[1], filling = 0.5, stat = -1)
    mft =   TBMFTModel(M, [t_flux, Ordering], [JParam], Function[InterQuarticToHopping], Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0))

    ##### Solve at each temperature, starting from the previous solution; F = E - TS per site
    Initial =   [0.3, 0.0]
    for (j, T) in enumerate(Ts)
        mft.model.T     =   T
        sc              =   SolveMFT!(mft, Initial ; max_iter = 300, tol = 1e-8)
        Initial         =   sc.VOuts[end]
        FreeEnergy[i, j]    =   GetMFTFreeEnergy(mft)
        (i, j) == (length(ϕs), 1) && append!(energies, mft.MFTEnergy)
    end
    println("ϕ/π = $(round(ϕ / π, digits = 3)):  F(T) = $(round.(FreeEnergy[i, :], digits = 6))")
end
println("Lowest free energy at ϕ/π = ", [round(ϕs[argmin(FreeEnergy[:, j])] / π, digits = 3) for j in eachindex(Ts)], " for T = ", Ts)

p0  =   plot(0:length(energies) - 1, energies, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}")
p1  =   plot(ϕs ./ π, FreeEnergy .- minimum(FreeEnergy, dims = 1), marker = :circle, lw = 2, label = permutedims([L"T = %$(T)J" for T in Ts]),
             xlabel = L"\phi / \pi", ylabel = L"F(\phi) - F_{\min}")
plot(p0, p1, layout = (1, 2), size = (900, 350), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, plot_title = L"\mathrm{Kagome\ spinon\ flux\ states}\ [\phi, \phi, \pi - 2\phi]")
