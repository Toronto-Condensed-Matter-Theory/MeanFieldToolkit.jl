### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ b89b0b98-3a34-47cf-8ccf-a4edbc9ea9d6
md"""
# Kagome lattice: Hubbard, Heisenberg and t-J mean-field theory

Unrestricted mean-field theory on the kagome lattice, using its three-site primitive cell, which fits q = 0 (120°) order.
Every spin channel on every site and nearest-neighbour bond is a free order parameter (and, for the t-J model,
every singlet and triplet pairing channel), so no ordering pattern is assumed. Expect: a paramagnet at small U and q = 0 120° order at large U; classical energy -0.25J and T_N = 0.25J for the Heisenberg model; no pairing in the t-J model at J = 0.3t.

The model builders live in `../common.jl`; `../MinimalExamples/Kagome*.jl` spell out the same calculations line by line.
"""

# ╔═╡ cd4a301d-b39d-4557-ad75-ad8889dde5c1
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))          # the Sample environment
    Pkg.instantiate()
    using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
    using LinearAlgebra, Logging, Plots, LaTeXStrings
    include(joinpath(@__DIR__, "..", "common.jl"))
    global_logger(ConsoleLogger(stderr, Logging.Warn))
end

# ╔═╡ 9ed3e2b0-169b-47a8-89a2-2ef19307c824
lattice = :kagome

# ╔═╡ ec1c0152-a101-470c-b5dc-1bd81fb248bd
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

# ╔═╡ 0f12f664-4be3-432e-acd5-a89e8858295c
let uc = new_cell(lattice), p = plot(aspect_ratio = 1, legend = false, grid = false, framestyle = :none)
    for bond in nn_bonds(uc)
        r1, r2  =   uc.basis[bond.base], uc.basis[bond.base] + bond_vector(uc, bond)
        plot!(p, [r1[1], r2[1]], [r1[2], r2[2]], color = :gray, lw = 2)
    end
    scatter!(p, first.(uc.basis), last.(uc.basis), ms = 9, series_annotations = text.(string.(eachindex(uc.basis)), :white, 8))
    p
end

# ╔═╡ 1d9495c1-341a-45c9-a282-e5842536a9de
md"""
## Hubbard model at half filling

``H = -t \sum_{\langle ij \rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}``, decoupled in every on-site channel (Hartree and Fock).
Each U starts from the lattice's classical spin pattern.
"""

# ╔═╡ 0ca3163e-a0e9-4a4a-9e78-f787932428cb
Us = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

# ╔═╡ 22912133-3b4e-499b-a9f5-a85b0e10bb64
hubbard = map(Us) do U
    mft =   hubbard_mft(lattice ; U = U)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (U = U, mft = mft, E = mft.MFTEnergy[end], m = 2max_moment(mft), pattern = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 51e17d58-1f05-4eb9-8c42-fcfc745d0e65
Markdown.parse(join(["| U/t | E/site | ``m = 2\\lvert\\langle S \\rangle\\rvert`` | pattern | converged |", "|---|---|---|---|---|",
["| $(r.U) | $(round(r.E, digits = 5)) | $(round(r.m, digits = 4)) | $(r.pattern) | $(r.converged) |" for r in hubbard]...], "\n"))

# ╔═╡ 5665f5a2-23ed-4dbe-8268-13d1b4c3921e
plot(Us, getproperty.(hubbard, :m), marker = :circle, lw = 2, legend = false, xlabel = L"U/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|",
title = L"\mathrm{Hubbard\ model,\ half\ filling}")

# ╔═╡ b66ec501-666d-49fc-a014-1e7bac5b9081
let r = hubbard[findfirst(r -> r.U == 4.0, hubbard)]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ U = 4t")
end

# ╔═╡ e6a52e41-f133-403b-8108-8ae64869d2aa
md"""
### Finite temperature

Raise ``T`` step by step at fixed U, starting each temperature from the previous solution. The moment vanishes at the
ordering temperature ``T_N``; ``F = E - TS`` is the free energy per site. (Hartree-Fock ``T_N`` grows like U, because
it tracks local-moment formation; the true ordering scale at large U is the exchange ``J = 4t^2/U``.)
"""

# ╔═╡ 3c7f464c-484a-4537-bb35-9150128fd635
U_T, Ts_hubbard = 8.0, range(0.05, 2.5, length = 25)

# ╔═╡ 10e5a511-fcea-435b-896c-d3e64ed51afe
hubbardT = let mft = hubbard_mft(lattice ; U = U_T), x = initial_guess(mft, lattice), m = Float64[], F = Float64[]
    for T in Ts_hubbard
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(m, 2max_moment(mft))
        push!(F, GetMFTFreeEnergy(mft))
    end
    (m = m, F = F, T_N = Ts_hubbard[something(findfirst(<(1e-3), m), length(m))])
end

# ╔═╡ 5360e134-e42b-4930-95b1-b2896c48c237
plot(plot(Ts_hubbard, hubbardT.m, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"m"),
plot(Ts_hubbard, hubbardT.F, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"F = E - TS"),
layout = (1, 2), size = (800, 320), legend = false, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
plot_title = L"U = %$(U_T)t:\ T_N \approx %$(round(hubbardT.T_N, digits = 2))t")

# ╔═╡ 3196c984-2cf7-4544-b256-d6998decaad5
md"""
## Heisenberg model with fermionic spinons

``J \sum_{\langle ij \rangle} \mathbf{S}_i \cdot \mathbf{S}_j`` with ``\mathbf{S} = \tfrac12 f^\dagger \boldsymbol{\sigma} f`` and one spinon per site (on average).
With every channel free, mean-field theory finds the classical ordered state. With the on-site channels switched off
(`onsite = false`) only the spinon hopping ``\chi`` remains: a spin-liquid ansatz. From the uniform start used here
it finds the uniform state; other starts find dimerized or trimerized states, which are lower in mean-field theory.
"""

# ╔═╡ ba4a4f9e-0fe0-4c4a-b6b0-7bc288fcc7b5
heisenberg = map((true, false)) do onsite
    mft =   heisenberg_mft(lattice ; onsite = onsite)
    SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (decoupling = onsite ? "all channels" : "spinon hopping only", mft = mft, E = mft.MFTEnergy[end],
     spin = round.(norm.(site_moments(mft).spin), digits = 4), χ = round.(abs.(getproperty.(bond_table(mft), :χ)), digits = 4))
end

# ╔═╡ 88f86969-7341-4a43-a3d4-03381b550245
Markdown.parse(join(["| decoupling | E/site | ``\\lvert\\langle S \\rangle\\rvert`` per site | ``\\lvert\\chi\\rvert`` per bond |", "|---|---|---|---|",
["| $(r.decoupling) | $(round(r.E, digits = 6)) | $(r.spin) | $(r.χ) |" for r in heisenberg]...], "\n"))

# ╔═╡ 24c2220b-ef9f-415d-8b14-73cb97428c47
let r = heisenberg[1]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \mathrm{all\ channels}")
end

# ╔═╡ bc84fc75-eedd-456b-b22e-da23766d6294
Ts_heisenberg = range(0.02, 0.7, length = 35)

# ╔═╡ 8a1ca8d3-abad-410f-9aab-fc284145c525
heisenbergT = let mft = heisenberg_mft(lattice), x = initial_guess(mft, lattice), S = Float64[]
    for T in Ts_heisenberg
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(S, max_moment(mft))
    end
    S
end

# ╔═╡ a31752ff-31df-442c-b528-7203143b848f
plot(Ts_heisenberg, heisenbergT, marker = :circle, lw = 2, label = "mean-field spinons", xlabel = L"T/J", ylabel = L"|\langle \mathbf{S} \rangle|",
     title = L"\mathrm{Ordering\ temperature:\ } z|\cos\theta| J/8 = 0.25J")
vline!([0.25], ls = :dash, color = :black, label = L"z|\cos\theta|J/8")

# ╔═╡ f1417ab9-8f45-4efb-b73a-e1bb4aa51029
md"""
## Slave-boson t-J model

``c_{i\sigma} = b_i^\dagger f_{i\sigma}`` with condensed bosons ``\langle b \rangle^2 = \delta``: the spinons hop with ``\delta t`` at filling
``1 - \delta`` per site, and ``J\, \mathbf{S}_i \cdot \mathbf{S}_j`` is decoupled in every channel: magnetization, hopping ``\chi``, singlet
and triplet pairing ``\Delta``. The start has singlet pairing ``\Delta \propto e^{2i\theta}`` on a bond at angle ``\theta``
(d-wave on the square lattice, d + id on the triangular and honeycomb lattices) and a weak classical spin pattern.
"""

# ╔═╡ b9893bb6-c342-467e-aeb5-71a05b28a380
δs, J_tJ = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], 0.3

# ╔═╡ 57a6ac63-3247-497b-891b-964a9f937fdc
tJ = map(δs) do δ
    mft =   tJ_mft(lattice ; δ = δ, J = J_tJ)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice ; m = 0.1) ; max_iter = 500, tol = 1e-8)
    (δ = δ, mft = mft, x = sc.VOuts[end], E = mft.MFTEnergy[end], Δ = pairing_amplitude(mft),
     pairing = pairing_symmetry(bond_table(mft)), magnetism = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 5bcd062e-6368-4205-8db4-9a7aee79ae91
Markdown.parse(join(["| δ | E/site | max ``\\lvert\\Delta_s\\rvert`` | pairing | magnetism | converged |", "|---|---|---|---|---|---|",
["| $(r.δ) | $(round(r.E, digits = 5)) | $(round(r.Δ, digits = 4)) | $(r.pairing) | $(r.magnetism) | $(r.converged) |" for r in tJ]...], "\n"))

# ╔═╡ 1e1ee260-b717-40eb-a608-3cabf725a50a
plot(δs, getproperty.(tJ, :Δ), marker = :circle, lw = 2, legend = false, xlabel = L"\delta", ylabel = L"\max |\Delta_s|",
title = L"t\textrm{-}J,\ J = %$(J_tJ)t,\ T = 0.002t")

# ╔═╡ 91dbca17-0e72-4132-810e-6490b4a7b497
let r = argmax(r -> r.converged ? r.Δ : 0.0, tJ)
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \delta = %$(r.δ)")
end

# ╔═╡ c4ec1cd4-3280-450b-bb6e-f431f3250cd4
md"""
### Pairing temperature

At the doping with the strongest (converged) pairing, raise ``T`` until ``\Delta`` vanishes: the spinon pairing temperature
``T_\mathrm{pair}``. Physical superconductivity also needs the bosons to condense (``T_\mathrm{BEC} \propto \delta t``), so the
superconducting ``T_c`` is the lower of the two.
"""

# ╔═╡ 6f2f603d-8052-4c88-9348-335e881aeff5
Ts_tJ = range(0.002, 0.05, length = 25)

# ╔═╡ 0b847dfa-93f6-4621-9142-0bc3fba7302e
tJT = let best = argmax(r -> r.converged ? r.Δ : 0.0, tJ), mft = tJ_mft(lattice ; δ = best.δ, J = J_tJ), x = best.x, Δ = Float64[]
    for T in Ts_tJ
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 500, tol = 1e-8).VOuts[end]
        push!(Δ, pairing_amplitude(mft))
    end
    (δ = best.δ, Δ = Δ, T_pair = best.Δ < 1e-4 ? NaN : Ts_tJ[something(findfirst(<(1e-4), Δ), length(Δ))])
end

# ╔═╡ 58e470ae-5f5a-4846-a782-56ce4960a8e8
plot(Ts_tJ, tJT.Δ, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\max |\Delta_s|",
title = L"\delta = %$(tJT.δ):\ T_\mathrm{pair} \approx %$(round(tJT.T_pair, digits = 4))t")

# ╔═╡ Cell order:
# ╟─b89b0b98-3a34-47cf-8ccf-a4edbc9ea9d6
# ╠═cd4a301d-b39d-4557-ad75-ad8889dde5c1
# ╠═9ed3e2b0-169b-47a8-89a2-2ef19307c824
# ╟─ec1c0152-a101-470c-b5dc-1bd81fb248bd
# ╠═0f12f664-4be3-432e-acd5-a89e8858295c
# ╟─1d9495c1-341a-45c9-a282-e5842536a9de
# ╠═0ca3163e-a0e9-4a4a-9e78-f787932428cb
# ╠═22912133-3b4e-499b-a9f5-a85b0e10bb64
# ╠═51e17d58-1f05-4eb9-8c42-fcfc745d0e65
# ╠═5665f5a2-23ed-4dbe-8268-13d1b4c3921e
# ╠═b66ec501-666d-49fc-a014-1e7bac5b9081
# ╟─e6a52e41-f133-403b-8108-8ae64869d2aa
# ╠═3c7f464c-484a-4537-bb35-9150128fd635
# ╠═10e5a511-fcea-435b-896c-d3e64ed51afe
# ╠═5360e134-e42b-4930-95b1-b2896c48c237
# ╟─3196c984-2cf7-4544-b256-d6998decaad5
# ╠═ba4a4f9e-0fe0-4c4a-b6b0-7bc288fcc7b5
# ╠═88f86969-7341-4a43-a3d4-03381b550245
# ╠═24c2220b-ef9f-415d-8b14-73cb97428c47
# ╠═bc84fc75-eedd-456b-b22e-da23766d6294
# ╠═8a1ca8d3-abad-410f-9aab-fc284145c525
# ╠═a31752ff-31df-442c-b528-7203143b848f
# ╟─f1417ab9-8f45-4efb-b73a-e1bb4aa51029
# ╠═b9893bb6-c342-467e-aeb5-71a05b28a380
# ╠═57a6ac63-3247-497b-891b-964a9f937fdc
# ╠═5bcd062e-6368-4205-8db4-9a7aee79ae91
# ╠═1e1ee260-b717-40eb-a608-3cabf725a50a
# ╠═91dbca17-0e72-4132-810e-6490b4a7b497
# ╟─c4ec1cd4-3280-450b-bb6e-f431f3250cd4
# ╠═6f2f603d-8052-4c88-9348-335e881aeff5
# ╠═0b847dfa-93f6-4621-9142-0bc3fba7302e
# ╠═58e470ae-5f5a-4846-a782-56ce4960a8e8
