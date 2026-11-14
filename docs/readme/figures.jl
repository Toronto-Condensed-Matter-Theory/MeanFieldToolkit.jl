#=
Regenerates the figures in the README (docs/src/assets/readme/). Run from the repository root:

    julia --project=docs/readme -e 'using Pkg; Pkg.instantiate()'     # once
    julia --project=docs/readme -t 8 docs/readme/figures.jl

Every figure is an unrestricted mean-field calculation built with the helpers in Sample/common.jl, which wrap the
scripts in Sample/MinimalExamples.
=#
include("style.jl")
include(joinpath(@__DIR__, "..", "..", "Sample", "common.jl"))
using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))         ##### hide the per-iteration log


##### ---------------------------- 1. Hubbard model: moment against U on three lattices ----------------------------
##### Half filling, T = 0.01t. Each U starts from the lattice's classical spin pattern.
##### Hartree-Fock finds Néel order at any U on the square lattice (perfect nesting), a finite U_c on the honeycomb
##### lattice (Dirac points), and 120° order above a larger U_c on the triangular lattice.
function hubbard_moment_vs_U()
    Us      =   collect(0.25:0.25:8.0)
    names   =   Dict(:square => "square", :honeycomb => "honeycomb", :triangular => "triangular")
    plt     =   plot(xlabel = L"U/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|", legend = :bottomright, xlims = (0, 8), ylims = (0, 1.0))
    for lattice in (:square, :honeycomb, :triangular)
        ms  =   map(Us) do U
            mft =   hubbard_mft(lattice ; U = U)
            SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
            2max_moment(mft)
        end
        println(rpad(lattice, 12), "m(U) = ", round.(ms, digits = 3))
        plot!(plt, Us, ms, label = names[lattice], marker = :circle, markersize = 3, markerstrokewidth = 0)
    end
    savefig_readme(plt, "hubbard_moment_vs_U")
end


##### ---------------------------- 2. Hubbard model: Néel moment against temperature ----------------------------
##### Square lattice at half filling. T is raised step by step, starting each temperature from the previous solution.
function hubbard_moment_vs_T()
    Ts      =   collect(range(0.02, 2.0, length = 50))
    plt     =   plot(xlabel = L"T/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|", legend = :topright, xlims = (0, 2.0), ylims = (0, 1.0))
    for U in (2.0, 4.0, 8.0)
        mft =   hubbard_mft(:square ; U = U)
        x   =   initial_guess(mft, :square)
        ms  =   map(Ts) do T
            mft.model.T =   T
            x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
            2max_moment(mft)
        end
        T_N =   Ts[something(findfirst(<(1e-3), ms), length(Ts))]
        println("U = $(U)t:  T_N ≈ $(round(T_N, digits = 3))t")
        plot!(plt, Ts, ms, label = L"U = %$(Int(U))t")
    end
    savefig_readme(plt, "hubbard_moment_vs_T")
end


##### ---------------------------- 3. Slave-boson t-J model: d-wave pairing against doping ----------------------------
##### Square lattice, J = 0.3t, T = 0.002t. Every spin, hopping, singlet and triplet pairing channel on every bond is free,
##### and the solution is d-wave at every doping shown. At low doping, antiferromagnetic order coexists and suppresses Δ.
function tJ_pairing_vs_doping()
    δs      =   collect(0.02:0.02:0.30)
    rows    =   map(δs) do δ
        mft =   tJ_mft(:square ; δ = δ, J = 0.3)
        sc  =   SolveMFT!(mft, initial_guess(mft, :square ; m = 0.1) ; max_iter = 500, tol = 1e-8)
        (Δ = pairing_amplitude(mft), symmetry = pairing_symmetry(bond_table(mft)), m = 2max_moment(mft),
         converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-6)
    end
    for (δ, r) in zip(δs, rows)
        println("δ = $(δ):  Δ = $(round(r.Δ, digits = 4)), $(r.symmetry), m = $(round(r.m, digits = 4)), converged = $(r.converged)")
    end
    Δ       =   getproperty.(rows, :Δ)
    plt     =   plot(δs, Δ, xlabel = "hole doping δ", ylabel = "d-wave pairing |Δ|",
                     legend = false, xlims = (0, 0.31), ylims = (0, 1.15 * maximum(Δ)), marker = :circle, markersize = 3, markerstrokewidth = 0)
    savefig_readme(plt, "tJ_pairing_vs_doping")
end


##### ---------------------------- 4. Kagome Heisenberg model: spinon flux states ----------------------------
##### Fermionic spinons with hopping of fixed pattern [ϕ, ϕ, π - 2ϕ] (flux through the two triangles and the hexagon)
##### on a 6-site cell; the amplitude is self-consistent. ϕ = 0 is the U(1) Dirac spin liquid, ϕ = π/2 the chiral spin
##### liquid. Same construction as Sample/MinimalExamples/KagomeFluxScan.jl, on a finer ϕ grid.
function kagome_flux_scan()
    sites   =   [[0.0, 0.0], [1.0, 0.0], [0.5, sqrt(3) / 2], [2.0, 0.0], [3.0, 0.0], [2.5, sqrt(3) / 2]]
    ϕs      =   collect(range(0.0, π / 2, length = 13))
    Ts      =   [0.001, 0.05, 0.1, 0.15]
    F       =   zeros(length(ϕs), length(Ts))
    for (i, ϕ) in enumerate(ϕs)
        uc  =   UnitCell([[4.0, 0.0], [1.0, sqrt(3)]], 2, 2)
        for site in sites
            AddBasisSite!(uc, site)
        end
        JParam  =   Param(1.0, 4)
        AddIsotropicBonds!(JParam, uc, 1.0, HEISENBERG_J, "Heisenberg Interaction")
        t_flux  =   Param(1.0, 2)
        for (base, target, offset, phase) in ((1, 2, [0, 0], ϕ / 3), (2, 3, [0, 0], ϕ / 3), (3, 1, [0, 0], ϕ / 3),
                                              (4, 5, [0, 0], ϕ / 3), (5, 6, [0, 0], ϕ / 3), (6, 4, [0, 0], ϕ / 3), (4, 2, [0, 0], ϕ / 3),
                                              (2, 6, [0, -1], π + ϕ / 3), (4, 6, [0, -1], -(π + ϕ / 3)), (5, 3, [1, -1], π + ϕ / 3),
                                              (5, 1, [1, 0], -(π + ϕ / 3)), (3, 1, [0, 1], ϕ / 3))
            AddAnisotropicBond!(t_flux, uc, base, target, offset, cis(phase) * SPIN[4], 1.0, "[ϕ, ϕ, π - 2ϕ] Hopping")
        end
        bz  =   filled_bz(uc, 33)
        H   =   Hamiltonian(uc, bz)
        DiagonalizeHamiltonian!(H)
        M   =   Model(uc, bz, H ; T = Ts[1], filling = 0.5, stat = -1)
        mft =   TBMFTModel(M, [t_flux], [JParam], Function[InterQuarticToHopping], Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0))
        x   =   [0.3]
        for (j, T) in enumerate(Ts)
            mft.model.T =   T
            x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
            F[i, j]     =   GetMFTFreeEnergy(mft)
        end
        println("ϕ/π = $(round(ϕ / π, digits = 3)):  F(T) = $(round.(F[i, :], digits = 6))")
    end
    plt     =   plot(xlabel = L"\mathrm{triangle\ flux}\ \phi/\pi", ylabel = L"[F(\phi) - F_{\min}]\ /\ 10^{-3}J", legend = :topright, xlims = (0, 0.5))
    for (j, T) in enumerate(Ts)
        plot!(plt, ϕs ./ π, 1e3 .* (F[:, j] .- minimum(F[:, j])), label = L"T = %$(T)J", marker = :circle, markersize = 3, markerstrokewidth = 0)
    end
    savefig_readme(plt, "kagome_flux_scan")
end


##### All figures by default; pass names to make only some, e.g. `figures.jl kagome_flux_scan`.
const FIGURES   =   (hubbard_moment_vs_U, hubbard_moment_vs_T, tJ_pairing_vs_doping, kagome_flux_scan)
for figure in FIGURES
    isempty(ARGS) || string(figure) in ARGS || continue
    println("== ", figure)
    @time figure()
end
