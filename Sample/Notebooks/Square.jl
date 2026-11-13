### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 1309c0e2-d419-4989-bbf1-2ef6cf24fde7
md"""
# Square lattice: Hubbard, Heisenberg and t-J mean-field theory

Unrestricted mean-field theory on the square lattice, using a √2 × √2 unit cell with two sites, so that Néel order fits.
Every spin channel on every site and nearest-neighbour bond is a free order parameter (and, for the t-J model,
every singlet and triplet pairing channel), so no ordering pattern is assumed. Expect: Néel order at every U > 0 (perfect nesting); classical Néel energy -0.5J and T_N = zJ/8 = 0.5J for the Heisenberg model; d-wave pairing in the t-J model.

The model builders live in `../common.jl`; `../MinimalExamples/Square*.jl` spell out the same calculations line by line.
"""

# ╔═╡ 260ad79c-1ece-4ec7-92f4-5a9a3a260b26
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))          # the Sample environment
    Pkg.instantiate()
    using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
    using LinearAlgebra, Logging, Plots, LaTeXStrings
    include(joinpath(@__DIR__, "..", "common.jl"))
    global_logger(ConsoleLogger(stderr, Logging.Warn))
end

# ╔═╡ ffa7f555-da02-4aa9-9ec1-868e3b925ce6
lattice = :square

# ╔═╡ 65251266-3597-4e0a-9bc3-f905a01dcb36
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

# ╔═╡ dc5329a4-1496-438e-98e1-332aa3ae1065
let uc = new_cell(lattice), p = plot(aspect_ratio = 1, legend = false, grid = false, framestyle = :none)
    for bond in nn_bonds(uc)
        r1, r2  =   uc.basis[bond.base], uc.basis[bond.base] + bond_vector(uc, bond)
        plot!(p, [r1[1], r2[1]], [r1[2], r2[2]], color = :gray, lw = 2)
    end
    scatter!(p, first.(uc.basis), last.(uc.basis), ms = 9, series_annotations = text.(string.(eachindex(uc.basis)), :white, 8))
    p
end

# ╔═╡ 27eb0446-999f-4205-986c-8cbab5273d37
md"""
## Hubbard model at half filling

``H = -t \sum_{\langle ij \rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}``, decoupled in every on-site channel (Hartree and Fock).
Each U starts from the lattice's classical spin pattern.
"""

# ╔═╡ ca5f8c3f-c583-4d69-afd6-0947890908bc
Us = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

# ╔═╡ 105bf492-4612-4b25-bb6c-bce5747d0ea1
hubbard = map(Us) do U
    mft =   hubbard_mft(lattice ; U = U)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (U = U, mft = mft, E = mft.MFTEnergy[end], m = 2max_moment(mft), pattern = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 1109b4a8-ddfd-4f61-a26c-c0d304a0394a
Markdown.parse(join(["| U/t | E/site | ``m = 2\\lvert\\langle S \\rangle\\rvert`` | pattern | converged |", "|---|---|---|---|---|",
["| $(r.U) | $(round(r.E, digits = 5)) | $(round(r.m, digits = 4)) | $(r.pattern) | $(r.converged) |" for r in hubbard]...], "\n"))

# ╔═╡ e4b8af0b-4f59-4fed-be7b-9cafdee33587
plot(Us, getproperty.(hubbard, :m), marker = :circle, lw = 2, legend = false, xlabel = L"U/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|",
title = L"\mathrm{Hubbard\ model,\ half\ filling}")

# ╔═╡ 240a404c-c162-43a1-9c97-48e2d8e812f6
let r = hubbard[findfirst(r -> r.U == 4.0, hubbard)]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ U = 4t")
end

# ╔═╡ 39d196a7-f339-48a0-906e-4d244d0822bb
md"""
### Finite temperature

Raise ``T`` step by step at fixed U, starting each temperature from the previous solution. The moment vanishes at the
ordering temperature ``T_N``; ``F = E - TS`` is the free energy per site. (Hartree-Fock ``T_N`` grows like U, because
it tracks local-moment formation; the true ordering scale at large U is the exchange ``J = 4t^2/U``.)
"""

# ╔═╡ d63b2359-2c45-4d37-8df5-f920813ab571
U_T, Ts_hubbard = 8.0, range(0.05, 2.5, length = 25)

# ╔═╡ 98eb48f4-75ff-4249-a477-a02ecce3cc7e
hubbardT = let mft = hubbard_mft(lattice ; U = U_T), x = initial_guess(mft, lattice), m = Float64[], F = Float64[]
    for T in Ts_hubbard
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(m, 2max_moment(mft))
        push!(F, GetMFTFreeEnergy(mft))
    end
    (m = m, F = F, T_N = Ts_hubbard[something(findfirst(<(1e-3), m), length(m))])
end

# ╔═╡ 1bd68dbe-c974-41fc-88d4-9eaf4cc3005a
plot(plot(Ts_hubbard, hubbardT.m, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"m"),
plot(Ts_hubbard, hubbardT.F, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"F = E - TS"),
layout = (1, 2), size = (800, 320), legend = false, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
plot_title = L"U = %$(U_T)t:\ T_N \approx %$(round(hubbardT.T_N, digits = 2))t")

# ╔═╡ 1dba0751-654e-4b25-9dce-d2ee00a76362
md"""
## Heisenberg model with fermionic spinons

``J \sum_{\langle ij \rangle} \mathbf{S}_i \cdot \mathbf{S}_j`` with ``\mathbf{S} = \tfrac12 f^\dagger \boldsymbol{\sigma} f`` and one spinon per site (on average).
With every channel free, mean-field theory finds the classical ordered state. With the on-site channels switched off
(`onsite = false`) only the spinon hopping ``\chi`` remains: a spin-liquid ansatz. From the uniform start used here
it finds the uniform state; other starts find dimerized or trimerized states, which are lower in mean-field theory.
"""

# ╔═╡ fa933c81-1b88-44af-9a54-2afae773f9c4
heisenberg = map((true, false)) do onsite
    mft =   heisenberg_mft(lattice ; onsite = onsite)
    SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (decoupling = onsite ? "all channels" : "spinon hopping only", mft = mft, E = mft.MFTEnergy[end],
     spin = round.(norm.(site_moments(mft).spin), digits = 4), χ = round.(abs.(getproperty.(bond_table(mft), :χ)), digits = 4))
end

# ╔═╡ 1626f9d4-6c72-41e5-8582-8a75bb859ee6
Markdown.parse(join(["| decoupling | E/site | ``\\lvert\\langle S \\rangle\\rvert`` per site | ``\\lvert\\chi\\rvert`` per bond |", "|---|---|---|---|",
["| $(r.decoupling) | $(round(r.E, digits = 6)) | $(r.spin) | $(r.χ) |" for r in heisenberg]...], "\n"))

# ╔═╡ 3b6c4c9d-ff87-46e6-9e81-4cc5788dcfa0
let r = heisenberg[1]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \mathrm{all\ channels}")
end

# ╔═╡ c6485712-a914-4472-8125-39539909042e
Ts_heisenberg = range(0.02, 0.7, length = 35)

# ╔═╡ d3833919-3732-4498-9422-3e316966af37
heisenbergT = let mft = heisenberg_mft(lattice), x = initial_guess(mft, lattice), S = Float64[]
    for T in Ts_heisenberg
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(S, max_moment(mft))
    end
    S
end

# ╔═╡ 3df60423-5651-4b74-8538-40712f104047
plot(Ts_heisenberg, heisenbergT, marker = :circle, lw = 2, label = "mean-field spinons", xlabel = L"T/J", ylabel = L"|\langle \mathbf{S} \rangle|",
     title = L"\mathrm{Ordering\ temperature:\ } z|\cos\theta| J/8 = 0.5J")
vline!([0.5], ls = :dash, color = :black, label = L"z|\cos\theta|J/8")

# ╔═╡ 9d531e90-2f57-43f6-9f26-44f9b3c1564f
md"""
## Slave-boson t-J model

``c_{i\sigma} = b_i^\dagger f_{i\sigma}`` with condensed bosons ``\langle b \rangle^2 = \delta``: the spinons hop with ``\delta t`` at filling
``1 - \delta`` per site, and ``J\, \mathbf{S}_i \cdot \mathbf{S}_j`` is decoupled in every channel: magnetization, hopping ``\chi``, singlet
and triplet pairing ``\Delta``. The start has singlet pairing ``\Delta \propto e^{2i\theta}`` on a bond at angle ``\theta``
(d-wave on the square lattice, d + id on the triangular and honeycomb lattices) and a weak classical spin pattern.
"""

# ╔═╡ e59e235f-a43b-4891-ae48-0214da635549
δs, J_tJ = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], 0.3

# ╔═╡ 532a61fd-88de-4026-bd42-b4b6d80c3752
tJ = map(δs) do δ
    mft =   tJ_mft(lattice ; δ = δ, J = J_tJ)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice ; m = 0.1) ; max_iter = 500, tol = 1e-8)
    (δ = δ, mft = mft, x = sc.VOuts[end], E = mft.MFTEnergy[end], Δ = pairing_amplitude(mft),
     pairing = pairing_symmetry(bond_table(mft)), magnetism = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 970d69e9-d470-447b-93d5-05b244905671
Markdown.parse(join(["| δ | E/site | max ``\\lvert\\Delta_s\\rvert`` | pairing | magnetism | converged |", "|---|---|---|---|---|---|",
["| $(r.δ) | $(round(r.E, digits = 5)) | $(round(r.Δ, digits = 4)) | $(r.pairing) | $(r.magnetism) | $(r.converged) |" for r in tJ]...], "\n"))

# ╔═╡ 128da3cc-0fb6-4fc2-93ae-bf3d4778ab69
plot(δs, getproperty.(tJ, :Δ), marker = :circle, lw = 2, legend = false, xlabel = L"\delta", ylabel = L"\max |\Delta_s|",
title = L"t\textrm{-}J,\ J = %$(J_tJ)t,\ T = 0.002t")

# ╔═╡ 2870de21-c31b-4830-8937-0c6dc1491811
let r = argmax(r -> r.converged ? r.Δ : 0.0, tJ)
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \delta = %$(r.δ)")
end

# ╔═╡ 922e66d2-9de1-486b-bddb-3e2dd069dc4f
md"""
### Pairing temperature

At the doping with the strongest (converged) pairing, raise ``T`` until ``\Delta`` vanishes: the spinon pairing temperature
``T_\mathrm{pair}``. Physical superconductivity also needs the bosons to condense (``T_\mathrm{BEC} \propto \delta t``), so the
superconducting ``T_c`` is the lower of the two.
"""

# ╔═╡ f25f5200-4fee-42ae-ae68-6681dda688e4
Ts_tJ = range(0.002, 0.05, length = 25)

# ╔═╡ 50d950bb-7977-465b-9b31-c2ed1d497b02
tJT = let best = argmax(r -> r.converged ? r.Δ : 0.0, tJ), mft = tJ_mft(lattice ; δ = best.δ, J = J_tJ), x = best.x, Δ = Float64[]
    for T in Ts_tJ
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 500, tol = 1e-8).VOuts[end]
        push!(Δ, pairing_amplitude(mft))
    end
    (δ = best.δ, Δ = Δ, T_pair = best.Δ < 1e-4 ? NaN : Ts_tJ[something(findfirst(<(1e-4), Δ), length(Δ))])
end

# ╔═╡ 665d8bc4-a10f-4efa-9b32-8c99b1343dee
plot(Ts_tJ, tJT.Δ, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\max |\Delta_s|",
title = L"\delta = %$(tJT.δ):\ T_\mathrm{pair} \approx %$(round(tJT.T_pair, digits = 4))t")

# ╔═╡ Cell order:
# ╟─1309c0e2-d419-4989-bbf1-2ef6cf24fde7
# ╠═260ad79c-1ece-4ec7-92f4-5a9a3a260b26
# ╠═ffa7f555-da02-4aa9-9ec1-868e3b925ce6
# ╟─65251266-3597-4e0a-9bc3-f905a01dcb36
# ╠═dc5329a4-1496-438e-98e1-332aa3ae1065
# ╟─27eb0446-999f-4205-986c-8cbab5273d37
# ╠═ca5f8c3f-c583-4d69-afd6-0947890908bc
# ╠═105bf492-4612-4b25-bb6c-bce5747d0ea1
# ╠═1109b4a8-ddfd-4f61-a26c-c0d304a0394a
# ╠═e4b8af0b-4f59-4fed-be7b-9cafdee33587
# ╠═240a404c-c162-43a1-9c97-48e2d8e812f6
# ╟─39d196a7-f339-48a0-906e-4d244d0822bb
# ╠═d63b2359-2c45-4d37-8df5-f920813ab571
# ╠═98eb48f4-75ff-4249-a477-a02ecce3cc7e
# ╠═1bd68dbe-c974-41fc-88d4-9eaf4cc3005a
# ╟─1dba0751-654e-4b25-9dce-d2ee00a76362
# ╠═fa933c81-1b88-44af-9a54-2afae773f9c4
# ╠═1626f9d4-6c72-41e5-8582-8a75bb859ee6
# ╠═3b6c4c9d-ff87-46e6-9e81-4cc5788dcfa0
# ╠═c6485712-a914-4472-8125-39539909042e
# ╠═d3833919-3732-4498-9422-3e316966af37
# ╠═3df60423-5651-4b74-8538-40712f104047
# ╟─9d531e90-2f57-43f6-9f26-44f9b3c1564f
# ╠═e59e235f-a43b-4891-ae48-0214da635549
# ╠═532a61fd-88de-4026-bd42-b4b6d80c3752
# ╠═970d69e9-d470-447b-93d5-05b244905671
# ╠═128da3cc-0fb6-4fc2-93ae-bf3d4778ab69
# ╠═2870de21-c31b-4830-8937-0c6dc1491811
# ╟─922e66d2-9de1-486b-bddb-3e2dd069dc4f
# ╠═f25f5200-4fee-42ae-ae68-6681dda688e4
# ╠═50d950bb-7977-465b-9b31-c2ed1d497b02
# ╠═665d8bc4-a10f-4efa-9b32-8c99b1343dee
