### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 56a4cca0-0242-4f2f-b434-53dd2dd5b0a6
md"""
# Honeycomb lattice: Hubbard, Heisenberg and t-J mean-field theory

Unrestricted mean-field theory on the honeycomb lattice, using its two-site primitive cell, which already fits Néel order.
Every spin channel on every site and nearest-neighbour bond is a free order parameter (and, for the t-J model,
every singlet and triplet pairing channel), so no ordering pattern is assumed. Expect: a Dirac semimetal up to U_c ≈ 2.2t, then Néel order; classical Néel energy -0.375J and T_N = 3J/8 for the Heisenberg model; chiral d + id pairing in the t-J model.

The model builders live in `../common.jl`; `../MinimalExamples/Honeycomb*.jl` spell out the same calculations line by line.
"""

# ╔═╡ 9bc35cd7-9ee2-4bb9-b0ec-3a02ad1f82a9
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))          # the Sample environment
    Pkg.instantiate()
    using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
    using LinearAlgebra, Logging, Plots, LaTeXStrings
    include(joinpath(@__DIR__, "..", "common.jl"))
    global_logger(ConsoleLogger(stderr, Logging.Warn))
end

# ╔═╡ a6ed5db0-00f9-400e-8a1d-bca6b2fc20e4
lattice = :honeycomb

# ╔═╡ 67fe5baa-ecdf-4f20-8942-787a2d2a56ee
md"""
## How the order parameters are built

One `Param` per site (or bond) and per matrix of a complete 2×2 basis: ``S^x, S^y, S^z, \mathbb{1}`` for
``\langle f^\dagger_a f_b\rangle`` and ``i\sigma^y, \sigma^\mu i\sigma^y`` for pairing ``\langle f_a f_b\rangle``.
These are the two loops in `common.jl`:

```julia
for site in eachindex(uc.basis), (name, mat) in ["Sx" => S[1], "Sy" => S[2], "Sz" => S[3], "n" => S[4]]
    channel = Param(1.0 + 0.0im, 2)
    AddAnisotropicBond!(channel, uc, site, site, [0, 0], mat, 0.0, "\$(name) on site \$(site)")
    push!(orders, channel)
end

nn = Param(1.0, 2)                                   # a scratch Param, just to list the bonds
AddIsotropicBonds!(nn, uc, 1.0, S[4], "nearest neighbours")
for bond in nn.unitBonds, (name, mat) in ["χx" => S[1], "χy" => S[2], "χz" => S[3], "χ" => S[4]]
    channel = Param(1.0 + 0.0im, 2)
    AddAnisotropicBond!(channel, uc, bond.base, bond.target, bond.offset, mat, 1.0, "\$(name) on bond …")
    push!(orders, channel)
end
```

The unit cell and its nearest-neighbour bonds (numbers label the sites):
"""

# ╔═╡ a2e54c58-a707-422e-89df-f7ae4d0f176e
let uc = new_cell(lattice), p = plot(aspect_ratio = 1, legend = false, grid = false, framestyle = :none)
    for bond in nn_bonds(uc)
        r1, r2  =   uc.basis[bond.base], uc.basis[bond.base] + bond_vector(uc, bond)
        plot!(p, [r1[1], r2[1]], [r1[2], r2[2]], color = :gray, lw = 2)
    end
    scatter!(p, first.(uc.basis), last.(uc.basis), ms = 9, series_annotations = text.(string.(eachindex(uc.basis)), :white, 8))
    p
end

# ╔═╡ 809f4b11-7d79-4389-8f87-ed3e5198c730
md"""
## Hubbard model at half filling

``H = -t \sum_{\langle ij \rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}``, decoupled in every on-site channel (Hartree and Fock).
Each U starts from the lattice's classical spin pattern.
"""

# ╔═╡ 16716801-dd15-4922-a0b7-4a95e8f320e7
Us = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

# ╔═╡ e630a17e-6230-498c-8c2d-c6aa94f50872
hubbard = map(Us) do U
    mft =   hubbard_mft(lattice ; U = U)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (U = U, mft = mft, E = mft.MFTEnergy[end], m = 2max_moment(mft), pattern = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ b74b601b-b3a8-4b25-92d8-bd67ef4330d3
Markdown.parse(join(["| U/t | E/site | ``m = 2\\lvert\\langle S \\rangle\\rvert`` | pattern | converged |", "|---|---|---|---|---|",
["| $(r.U) | $(round(r.E, digits = 5)) | $(round(r.m, digits = 4)) | $(r.pattern) | $(r.converged) |" for r in hubbard]...], "\n"))

# ╔═╡ a1579211-4c6d-4fa9-aab6-30734429d3d4
plot(Us, getproperty.(hubbard, :m), marker = :circle, lw = 2, legend = false, xlabel = L"U/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|",
title = L"\mathrm{Hubbard\ model,\ half\ filling}")

# ╔═╡ 5d1677c2-4034-4022-871f-53d4c42d8e68
let r = hubbard[findfirst(r -> r.U == 4.0, hubbard)]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ U = 4t")
end

# ╔═╡ 1711da28-eee2-4f37-9fb0-5bd1f6687110
md"""
### Finite temperature

Raise ``T`` step by step at fixed U, starting each temperature from the previous solution. The moment vanishes at the
ordering temperature ``T_N``; ``F = E - TS`` is the free energy per site. (Hartree-Fock ``T_N`` grows like U, because
it tracks local-moment formation; the true ordering scale at large U is the exchange ``J = 4t^2/U``.)
"""

# ╔═╡ 997234b0-2cc1-40bb-be4c-1175e74fa993
U_T, Ts_hubbard = 8.0, range(0.05, 2.5, length = 25)

# ╔═╡ 8c8f3b71-f25f-41f2-8f3b-9f6e3ce84146
hubbardT = let mft = hubbard_mft(lattice ; U = U_T), x = initial_guess(mft, lattice), m = Float64[], F = Float64[]
    for T in Ts_hubbard
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(m, 2max_moment(mft))
        push!(F, GetMFTFreeEnergy(mft))
    end
    (m = m, F = F, T_N = Ts_hubbard[something(findfirst(<(1e-3), m), length(m))])
end

# ╔═╡ 99b12f54-cb66-4e30-b54f-406ee440b9c6
plot(plot(Ts_hubbard, hubbardT.m, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"m"),
plot(Ts_hubbard, hubbardT.F, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"F = E - TS"),
layout = (1, 2), size = (800, 320), legend = false, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
plot_title = L"U = %$(U_T)t:\ T_N \approx %$(round(hubbardT.T_N, digits = 2))t")

# ╔═╡ 5a32fcf6-485f-4af8-a3af-8167d9304af5
md"""
## Heisenberg model with fermionic spinons

``J \sum_{\langle ij \rangle} \mathbf{S}_i \cdot \mathbf{S}_j`` with ``\mathbf{S} = \tfrac12 f^\dagger \boldsymbol{\sigma} f`` and one spinon per site (on average).
With every channel free, mean-field theory finds the classical ordered state. With the on-site channels switched off
(`onsite = false`) only the spinon hopping ``\chi`` remains: a spin-liquid ansatz. From the uniform start used here
it finds the uniform state; other starts find dimerized or trimerized states, which are lower in mean-field theory.
"""

# ╔═╡ 2596f3f4-fd79-4160-9e28-5562ce451021
heisenberg = map((true, false)) do onsite
    mft =   heisenberg_mft(lattice ; onsite = onsite)
    SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (decoupling = onsite ? "all channels" : "spinon hopping only", mft = mft, E = mft.MFTEnergy[end],
     spin = round.(norm.(site_moments(mft).spin), digits = 4), χ = round.(abs.(getproperty.(bond_table(mft), :χ)), digits = 4))
end

# ╔═╡ 9a5f031c-2032-4518-be00-fd4918157388
Markdown.parse(join(["| decoupling | E/site | ``\\lvert\\langle S \\rangle\\rvert`` per site | ``\\lvert\\chi\\rvert`` per bond |", "|---|---|---|---|",
["| $(r.decoupling) | $(round(r.E, digits = 6)) | $(r.spin) | $(r.χ) |" for r in heisenberg]...], "\n"))

# ╔═╡ 5de0513c-3b8c-46c5-aca3-642c0953ae16
let r = heisenberg[1]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \mathrm{all\ channels}")
end

# ╔═╡ 7fc6c985-e1ec-46e5-96a6-8661f3dfe652
Ts_heisenberg = range(0.02, 0.7, length = 35)

# ╔═╡ 2af506ad-efd4-4c0c-8a42-e633291d654f
heisenbergT = let mft = heisenberg_mft(lattice), x = initial_guess(mft, lattice), S = Float64[]
    for T in Ts_heisenberg
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(S, max_moment(mft))
    end
    S
end

# ╔═╡ 440a504f-749c-47dc-b845-05e9644e2d34
plot(Ts_heisenberg, heisenbergT, marker = :circle, lw = 2, label = "mean-field spinons", xlabel = L"T/J", ylabel = L"|\langle \mathbf{S} \rangle|",
     title = L"\mathrm{Ordering\ temperature:\ } z|\cos\theta| J/8 = 0.375J")
vline!([0.375], ls = :dash, color = :black, label = L"z|\cos\theta|J/8")

# ╔═╡ 7f064604-2d27-4218-a1b6-67af953cdea7
md"""
## Slave-boson t-J model

``c_{i\sigma} = b_i^\dagger f_{i\sigma}`` with condensed bosons ``\langle b \rangle^2 = \delta``: the spinons hop with ``\delta t`` at filling
``1 - \delta`` per site, and ``J\, \mathbf{S}_i \cdot \mathbf{S}_j`` is decoupled in every channel: magnetization, hopping ``\chi``, singlet
and triplet pairing ``\Delta``. The start has singlet pairing ``\Delta \propto e^{2i\theta}`` on a bond at angle ``\theta``
(d-wave on the square lattice, d + id on the triangular and honeycomb lattices) and a weak classical spin pattern.
"""

# ╔═╡ e86359ae-a892-4311-a8ee-d861ea405103
δs, J_tJ = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], 0.3

# ╔═╡ a41f3a85-ad3a-430a-9218-0e7f22857fad
tJ = map(δs) do δ
    mft =   tJ_mft(lattice ; δ = δ, J = J_tJ)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice ; m = 0.1) ; max_iter = 500, tol = 1e-8)
    (δ = δ, mft = mft, x = sc.VOuts[end], E = mft.MFTEnergy[end], Δ = pairing_amplitude(mft),
     pairing = pairing_symmetry(bond_table(mft)), magnetism = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ c3df0bc0-169e-4082-9898-ff0e8ef0cbac
Markdown.parse(join(["| δ | E/site | max ``\\lvert\\Delta_s\\rvert`` | pairing | magnetism | converged |", "|---|---|---|---|---|---|",
["| $(r.δ) | $(round(r.E, digits = 5)) | $(round(r.Δ, digits = 4)) | $(r.pairing) | $(r.magnetism) | $(r.converged) |" for r in tJ]...], "\n"))

# ╔═╡ 7b05e39e-9913-4d3f-b853-6f8c66faaaf6
plot(δs, getproperty.(tJ, :Δ), marker = :circle, lw = 2, legend = false, xlabel = L"\delta", ylabel = L"\max |\Delta_s|",
title = L"t\textrm{-}J,\ J = %$(J_tJ)t,\ T = 0.002t")

# ╔═╡ 03a457d3-7cd4-44ae-9642-a698a2ca21bb
let r = argmax(r -> r.converged ? r.Δ : 0.0, tJ)
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \delta = %$(r.δ)")
end

# ╔═╡ 36c350d8-d6cf-4b95-b84e-e34677ae007d
md"""
### Pairing temperature

At the doping with the strongest (converged) pairing, raise ``T`` until ``\Delta`` vanishes: the spinon pairing temperature
``T_\mathrm{pair}``. Physical superconductivity also needs the bosons to condense (``T_\mathrm{BEC} \propto \delta t``), so the
superconducting ``T_c`` is the lower of the two.
"""

# ╔═╡ 1de3305e-08ab-4ac5-985d-196e639387e4
Ts_tJ = range(0.002, 0.05, length = 25)

# ╔═╡ 9422aa08-a89c-4b2d-b5f1-be40878885f1
tJT = let best = argmax(r -> r.converged ? r.Δ : 0.0, tJ), mft = tJ_mft(lattice ; δ = best.δ, J = J_tJ), x = best.x, Δ = Float64[]
    for T in Ts_tJ
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 500, tol = 1e-8).VOuts[end]
        push!(Δ, pairing_amplitude(mft))
    end
    (δ = best.δ, Δ = Δ, T_pair = best.Δ < 1e-4 ? NaN : Ts_tJ[something(findfirst(<(1e-4), Δ), length(Δ))])
end

# ╔═╡ db08817e-1553-46f1-b2dc-3ff272a91050
plot(Ts_tJ, tJT.Δ, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\max |\Delta_s|",
title = L"\delta = %$(tJT.δ):\ T_\mathrm{pair} \approx %$(round(tJT.T_pair, digits = 4))t")

# ╔═╡ Cell order:
# ╟─56a4cca0-0242-4f2f-b434-53dd2dd5b0a6
# ╠═9bc35cd7-9ee2-4bb9-b0ec-3a02ad1f82a9
# ╠═a6ed5db0-00f9-400e-8a1d-bca6b2fc20e4
# ╟─67fe5baa-ecdf-4f20-8942-787a2d2a56ee
# ╠═a2e54c58-a707-422e-89df-f7ae4d0f176e
# ╟─809f4b11-7d79-4389-8f87-ed3e5198c730
# ╠═16716801-dd15-4922-a0b7-4a95e8f320e7
# ╠═e630a17e-6230-498c-8c2d-c6aa94f50872
# ╠═b74b601b-b3a8-4b25-92d8-bd67ef4330d3
# ╠═a1579211-4c6d-4fa9-aab6-30734429d3d4
# ╠═5d1677c2-4034-4022-871f-53d4c42d8e68
# ╟─1711da28-eee2-4f37-9fb0-5bd1f6687110
# ╠═997234b0-2cc1-40bb-be4c-1175e74fa993
# ╠═8c8f3b71-f25f-41f2-8f3b-9f6e3ce84146
# ╠═99b12f54-cb66-4e30-b54f-406ee440b9c6
# ╟─5a32fcf6-485f-4af8-a3af-8167d9304af5
# ╠═2596f3f4-fd79-4160-9e28-5562ce451021
# ╠═9a5f031c-2032-4518-be00-fd4918157388
# ╠═5de0513c-3b8c-46c5-aca3-642c0953ae16
# ╠═7fc6c985-e1ec-46e5-96a6-8661f3dfe652
# ╠═2af506ad-efd4-4c0c-8a42-e633291d654f
# ╠═440a504f-749c-47dc-b845-05e9644e2d34
# ╟─7f064604-2d27-4218-a1b6-67af953cdea7
# ╠═e86359ae-a892-4311-a8ee-d861ea405103
# ╠═a41f3a85-ad3a-430a-9218-0e7f22857fad
# ╠═c3df0bc0-169e-4082-9898-ff0e8ef0cbac
# ╠═7b05e39e-9913-4d3f-b853-6f8c66faaaf6
# ╠═03a457d3-7cd4-44ae-9642-a698a2ca21bb
# ╟─36c350d8-d6cf-4b95-b84e-e34677ae007d
# ╠═1de3305e-08ab-4ac5-985d-196e639387e4
# ╠═9422aa08-a89c-4b2d-b5f1-be40878885f1
# ╠═db08817e-1553-46f1-b2dc-3ff272a91050
