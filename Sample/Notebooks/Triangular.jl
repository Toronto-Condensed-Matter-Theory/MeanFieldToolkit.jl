### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 50234373-3061-4f28-a7f2-9c2ac56c3a32
md"""
# Triangular lattice: Hubbard, Heisenberg and t-J mean-field theory

Unrestricted mean-field theory on the triangular lattice, using a √3 × √3 unit cell with three sites, so that 120° order fits.
Every spin channel on every site and nearest-neighbour bond is a free order parameter (and, for the t-J model,
every singlet and triplet pairing channel), so no ordering pattern is assumed. Expect: a paramagnet at small U and 120° order at large U; classical 120° energy -0.375J and T_N = z|cos θ|J/8 = 0.375J for the Heisenberg model; chiral d + id pairing in the t-J model.

The model builders live in `../common.jl`; `../MinimalExamples/Triangular*.jl` spell out the same calculations line by line.
"""

# ╔═╡ 752491b6-658a-4bc5-b2e2-0b9a933253c7
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))          # the Sample environment
    Pkg.instantiate()
    using MeanFieldToolkit, TightBindingToolkit, FixedPointToolkit
    using LinearAlgebra, Logging, Plots, LaTeXStrings
    include(joinpath(@__DIR__, "..", "common.jl"))
    global_logger(ConsoleLogger(stderr, Logging.Warn))
end

# ╔═╡ 77c31366-3eb3-4bea-b998-ab8833e21888
lattice = :triangular

# ╔═╡ 5f231dfd-939a-4eb1-9352-2de2894b8ff7
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

# ╔═╡ 3c5d7ed0-0d87-4333-bbc6-43683af0e6b0
let uc = new_cell(lattice), p = plot(aspect_ratio = 1, legend = false, grid = false, framestyle = :none)
    for bond in nn_bonds(uc)
        r1, r2  =   uc.basis[bond.base], uc.basis[bond.base] + bond_vector(uc, bond)
        plot!(p, [r1[1], r2[1]], [r1[2], r2[2]], color = :gray, lw = 2)
    end
    scatter!(p, first.(uc.basis), last.(uc.basis), ms = 9, series_annotations = text.(string.(eachindex(uc.basis)), :white, 8))
    p
end

# ╔═╡ 7b6e68ac-90ed-4b5b-8707-b879d212aaf0
md"""
## Hubbard model at half filling

``H = -t \sum_{\langle ij \rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} + U \sum_i n_{i\uparrow} n_{i\downarrow}``, decoupled in every on-site channel (Hartree and Fock).
Each U starts from the lattice's classical spin pattern.
"""

# ╔═╡ 4a075c29-67c0-4bcb-92a6-b21b7adb1ffb
Us = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

# ╔═╡ 76feec33-880d-4a5f-a6d8-40c4e35f40f0
hubbard = map(Us) do U
    mft =   hubbard_mft(lattice ; U = U)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (U = U, mft = mft, E = mft.MFTEnergy[end], m = 2max_moment(mft), pattern = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 8b1af1c0-3355-4646-8836-a198564e3976
Markdown.parse(join(["| U/t | E/site | ``m = 2\\lvert\\langle S \\rangle\\rvert`` | pattern | converged |", "|---|---|---|---|---|",
["| $(r.U) | $(round(r.E, digits = 5)) | $(round(r.m, digits = 4)) | $(r.pattern) | $(r.converged) |" for r in hubbard]...], "\n"))

# ╔═╡ c5b4a48a-730f-4d5f-a83a-2b7d842ca4c8
plot(Us, getproperty.(hubbard, :m), marker = :circle, lw = 2, legend = false, xlabel = L"U/t", ylabel = L"m = 2|\langle \mathbf{S} \rangle|",
title = L"\mathrm{Hubbard\ model,\ half\ filling}")

# ╔═╡ 3bd87f18-37f4-492c-881e-7ae45e90537b
let r = hubbard[findfirst(r -> r.U == 4.0, hubbard)]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ U = 4t")
end

# ╔═╡ f1d03f24-e750-4103-8077-a2b4514c441e
md"""
### Finite temperature

Raise ``T`` step by step at fixed U, starting each temperature from the previous solution. The moment vanishes at the
ordering temperature ``T_N``; ``F = E - TS`` is the free energy per site. (Hartree-Fock ``T_N`` grows like U, because
it tracks local-moment formation; the true ordering scale at large U is the exchange ``J = 4t^2/U``.)
"""

# ╔═╡ 02e4094a-6cc6-4326-b263-4d20a28a96e1
U_T, Ts_hubbard = 8.0, range(0.05, 2.5, length = 25)

# ╔═╡ 603b0f66-8735-4b56-a932-f3a82bf1a76d
hubbardT = let mft = hubbard_mft(lattice ; U = U_T), x = initial_guess(mft, lattice), m = Float64[], F = Float64[]
    for T in Ts_hubbard
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(m, 2max_moment(mft))
        push!(F, GetMFTFreeEnergy(mft))
    end
    (m = m, F = F, T_N = Ts_hubbard[something(findfirst(<(1e-3), m), length(m))])
end

# ╔═╡ bae0c51c-5618-4235-934b-b2f97455cd6e
plot(plot(Ts_hubbard, hubbardT.m, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"m"),
plot(Ts_hubbard, hubbardT.F, marker = :circle, lw = 2, xlabel = L"T/t", ylabel = L"F = E - TS"),
layout = (1, 2), size = (800, 320), legend = false, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
plot_title = L"U = %$(U_T)t:\ T_N \approx %$(round(hubbardT.T_N, digits = 2))t")

# ╔═╡ 100e5c78-a8dc-4692-9a9e-0fb4b3f12679
md"""
## Heisenberg model with fermionic spinons

``J \sum_{\langle ij \rangle} \mathbf{S}_i \cdot \mathbf{S}_j`` with ``\mathbf{S} = \tfrac12 f^\dagger \boldsymbol{\sigma} f`` and one spinon per site (on average).
With every channel free, mean-field theory finds the classical ordered state. With the on-site channels switched off
(`onsite = false`) only the spinon hopping ``\chi`` remains: a spin-liquid ansatz. From the uniform start used here
it finds the uniform state; other starts find dimerized or trimerized states, which are lower in mean-field theory.
"""

# ╔═╡ c0410d0b-2f39-4e89-96ff-c599f77e035a
heisenberg = map((true, false)) do onsite
    mft =   heisenberg_mft(lattice ; onsite = onsite)
    SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-8)
    (decoupling = onsite ? "all channels" : "spinon hopping only", mft = mft, E = mft.MFTEnergy[end],
     spin = round.(norm.(site_moments(mft).spin), digits = 4), χ = round.(abs.(getproperty.(bond_table(mft), :χ)), digits = 4))
end

# ╔═╡ acf34363-baea-47cb-913b-2befb5864a7c
Markdown.parse(join(["| decoupling | E/site | ``\\lvert\\langle S \\rangle\\rvert`` per site | ``\\lvert\\chi\\rvert`` per bond |", "|---|---|---|---|",
["| $(r.decoupling) | $(round(r.E, digits = 6)) | $(r.spin) | $(r.χ) |" for r in heisenberg]...], "\n"))

# ╔═╡ 24e83733-105c-416d-b7c1-df4bc784cc4a
let r = heisenberg[1]
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \mathrm{all\ channels}")
end

# ╔═╡ 242727bc-8411-438b-85e6-2bbc3212d072
Ts_heisenberg = range(0.02, 0.7, length = 35)

# ╔═╡ c1d57167-813c-4002-b753-631bc7473dcf
heisenbergT = let mft = heisenberg_mft(lattice), x = initial_guess(mft, lattice), S = Float64[]
    for T in Ts_heisenberg
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 300, tol = 1e-8).VOuts[end]
        push!(S, max_moment(mft))
    end
    S
end

# ╔═╡ 43a556bb-3c41-48e5-bd3b-a99a7b4e31e3
plot(Ts_heisenberg, heisenbergT, marker = :circle, lw = 2, label = "mean-field spinons", xlabel = L"T/J", ylabel = L"|\langle \mathbf{S} \rangle|",
     title = L"\mathrm{Ordering\ temperature:\ } z|\cos\theta| J/8 = 0.375J")
vline!([0.375], ls = :dash, color = :black, label = L"z|\cos\theta|J/8")

# ╔═╡ 43606ff8-a1a4-44c4-92e9-7ffa5d04c478
md"""
## Slave-boson t-J model

``c_{i\sigma} = b_i^\dagger f_{i\sigma}`` with condensed bosons ``\langle b \rangle^2 = \delta``: the spinons hop with ``\delta t`` at filling
``1 - \delta`` per site, and ``J\, \mathbf{S}_i \cdot \mathbf{S}_j`` is decoupled in every channel: magnetization, hopping ``\chi``, singlet
and triplet pairing ``\Delta``. The start has singlet pairing ``\Delta \propto e^{2i\theta}`` on a bond at angle ``\theta``
(d-wave on the square lattice, d + id on the triangular and honeycomb lattices) and a weak classical spin pattern.
"""

# ╔═╡ 29c00b09-919f-44dc-a35c-c0b6b8b0cf4e
δs, J_tJ = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05], 0.3

# ╔═╡ 9ab64cfa-fa87-4f03-9b81-edc6a494af89
tJ = map(δs) do δ
    mft =   tJ_mft(lattice ; δ = δ, J = J_tJ)
    sc  =   SolveMFT!(mft, initial_guess(mft, lattice ; m = 0.1) ; max_iter = 500, tol = 1e-8)
    (δ = δ, mft = mft, x = sc.VOuts[end], E = mft.MFTEnergy[end], Δ = pairing_amplitude(mft),
     pairing = pairing_symmetry(bond_table(mft)), magnetism = magnetic_pattern(site_moments(mft).spin),
     converged = maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8)
end

# ╔═╡ 1abd3680-6717-4540-9049-f420f957f951
Markdown.parse(join(["| δ | E/site | max ``\\lvert\\Delta_s\\rvert`` | pairing | magnetism | converged |", "|---|---|---|---|---|---|",
["| $(r.δ) | $(round(r.E, digits = 5)) | $(round(r.Δ, digits = 4)) | $(r.pairing) | $(r.magnetism) | $(r.converged) |" for r in tJ]...], "\n"))

# ╔═╡ d72563c2-71e0-4455-8253-346a2e14ce21
plot(δs, getproperty.(tJ, :Δ), marker = :circle, lw = 2, legend = false, xlabel = L"\delta", ylabel = L"\max |\Delta_s|",
title = L"t\textrm{-}J,\ J = %$(J_tJ)t,\ T = 0.002t")

# ╔═╡ e3b0edea-bd11-4b95-9e56-551482943c87
let r = argmax(r -> r.converged ? r.Δ : 0.0, tJ)
    plot(0:length(r.mft.MFTEnergy) - 1, r.mft.MFTEnergy, marker = :circle, lw = 2, legend = false, xlabel = L"\mathrm{iteration}", ylabel = L"E/\mathrm{site}",
         title = L"\mathrm{Convergence:}\ \delta = %$(r.δ)")
end

# ╔═╡ 6cb384e2-5dd1-4f50-97c9-974a0202b628
md"""
### Pairing temperature

At the doping with the strongest (converged) pairing, raise ``T`` until ``\Delta`` vanishes: the spinon pairing temperature
``T_\mathrm{pair}``. Physical superconductivity also needs the bosons to condense (``T_\mathrm{BEC} \propto \delta t``), so the
superconducting ``T_c`` is the lower of the two.
"""

# ╔═╡ 033cf754-3c8b-49ab-a75b-893df241db33
Ts_tJ = range(0.002, 0.05, length = 25)

# ╔═╡ 63d70416-3f34-4830-9e9e-1197089660c7
tJT = let best = argmax(r -> r.converged ? r.Δ : 0.0, tJ), mft = tJ_mft(lattice ; δ = best.δ, J = J_tJ), x = best.x, Δ = Float64[]
    for T in Ts_tJ
        mft.model.T =   T
        x           =   SolveMFT!(mft, x ; max_iter = 500, tol = 1e-8).VOuts[end]
        push!(Δ, pairing_amplitude(mft))
    end
    (δ = best.δ, Δ = Δ, T_pair = best.Δ < 1e-4 ? NaN : Ts_tJ[something(findfirst(<(1e-4), Δ), length(Δ))])
end

# ╔═╡ b6f13f70-5d21-4113-83e0-29b996a16723
plot(Ts_tJ, tJT.Δ, marker = :circle, lw = 2, legend = false, xlabel = L"T/t", ylabel = L"\max |\Delta_s|",
title = L"\delta = %$(tJT.δ):\ T_\mathrm{pair} \approx %$(round(tJT.T_pair, digits = 4))t")

# ╔═╡ Cell order:
# ╟─50234373-3061-4f28-a7f2-9c2ac56c3a32
# ╠═752491b6-658a-4bc5-b2e2-0b9a933253c7
# ╠═77c31366-3eb3-4bea-b998-ab8833e21888
# ╟─5f231dfd-939a-4eb1-9352-2de2894b8ff7
# ╠═3c5d7ed0-0d87-4333-bbc6-43683af0e6b0
# ╟─7b6e68ac-90ed-4b5b-8707-b879d212aaf0
# ╠═4a075c29-67c0-4bcb-92a6-b21b7adb1ffb
# ╠═76feec33-880d-4a5f-a6d8-40c4e35f40f0
# ╠═8b1af1c0-3355-4646-8836-a198564e3976
# ╠═c5b4a48a-730f-4d5f-a83a-2b7d842ca4c8
# ╠═3bd87f18-37f4-492c-881e-7ae45e90537b
# ╟─f1d03f24-e750-4103-8077-a2b4514c441e
# ╠═02e4094a-6cc6-4326-b263-4d20a28a96e1
# ╠═603b0f66-8735-4b56-a932-f3a82bf1a76d
# ╠═bae0c51c-5618-4235-934b-b2f97455cd6e
# ╟─100e5c78-a8dc-4692-9a9e-0fb4b3f12679
# ╠═c0410d0b-2f39-4e89-96ff-c599f77e035a
# ╠═acf34363-baea-47cb-913b-2befb5864a7c
# ╠═24e83733-105c-416d-b7c1-df4bc784cc4a
# ╠═242727bc-8411-438b-85e6-2bbc3212d072
# ╠═c1d57167-813c-4002-b753-631bc7473dcf
# ╠═43a556bb-3c41-48e5-bd3b-a99a7b4e31e3
# ╟─43606ff8-a1a4-44c4-92e9-7ffa5d04c478
# ╠═29c00b09-919f-44dc-a35c-c0b6b8b0cf4e
# ╠═9ab64cfa-fa87-4f03-9b81-edc6a494af89
# ╠═1abd3680-6717-4540-9049-f420f957f951
# ╠═d72563c2-71e0-4455-8253-346a2e14ce21
# ╠═e3b0edea-bd11-4b95-9e56-551482943c87
# ╟─6cb384e2-5dd1-4f50-97c9-974a0202b628
# ╠═033cf754-3c8b-49ab-a75b-893df241db33
# ╠═63d70416-3f34-4830-9e9e-1197089660c7
# ╠═b6f13f70-5d21-4113-83e0-29b996a16723
