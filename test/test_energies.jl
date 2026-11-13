##### Independent checks of the mean-field energy (GetMFTEnergy).
#####
##### 1. Free fermions: the energy must equal the sum of occupied band energies.
##### 2. Hellmann-Feynman: at self-consistency the Hartree-Fock(-Bogoliubov) energy is stationary, so for an interaction of
#####    strength λ, dE/dλ = ⟨H_λ⟩/λ, where ⟨H_λ⟩ = ½⟨V_MF⟩ is the interaction energy. A mistake in either the
#####    kinetic or the double-counting bookkeeping breaks this identity (before v2.3 every inter-site and pairing term in
#####    ⟨H_MF⟩ was counted once instead of twice, and on-site pairing double counting was halved).

##### ½⟨V_MF⟩ per site, from the decompositions
function interaction_energy(mft::TBMFTModel)
    lookup  =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(mft.HoppingOrders))
    E       =   0.0
    for (i, int) in enumerate(mft.Interactions), (key, U) in Lookup([int])
        ex  =   GetBondDictionary(lookup, key, 2)
        E   +=  GetMFTBondEnergies(ex, mft.MFTDecomposition[i](U, ex), mft.model.uc ; scaling = mft.MFTScaling)
    end
    return E / length(mft.model.uc.basis)
end

function interaction_energy(mft::BdGMFTModel)
    hl  =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(mft.HoppingOrders))
    pl  =   Dict{Tuple, Matrix{ComplexF64}}(Lookup(mft.PairingOrders))
    E   =   0.0
    for (i, int) in enumerate(mft.Interactions), (key, U) in Lookup([int])
        h   =   GetBondDictionary(hl, key, 2)
        p   =   GetBondDictionary(pl, key, 2)
        E   +=  GetMFTBondEnergies(h, mft.HoppingDecomposition[i](U, h), mft.model.uc_hop ; scaling = mft.HoppingScaling)
        E   +=  GetMFTBondEnergies(p, mft.PairingDecomposition[i](U, p), mft.model.uc_pair ; scaling = mft.PairingScaling, onsite_weight = 1.0)
    end
    return E / length(mft.model.uc_hop.basis)
end

##### dE/dλ by central differences (each side started from the converged state at λ, to stay on the same branch) and ⟨H_λ⟩/λ
function hellmann_feynman(make, λ, x0 ; h = 1e-3 * λ)
    mft     =   make(λ)
    sc      =   SolveMFT!(mft, x0 ; max_iter = 500, tol = 1e-11)
    x       =   sc.VOuts[end]
    energy(μ)   =   (m = make(μ); SolveMFT!(m, copy(x) ; max_iter = 500, tol = 1e-11); m.MFTEnergy[end])
    return (energy(λ + h) - energy(λ - h)) / (2h), interaction_energy(mft) / λ, maximum(abs.(sc.VOuts[end] - sc.VIns[end]))
end

##### square lattice t-J without pairing, magnetic (Hartree) channels off: only inter-site Fock decoupling of J
function tJ_fock(J ; δ = 0.1, kSize = 15)
    uc      =   new_cell(:square)
    add_nn_hopping!(uc, δ)
    JParam  =   nn_interaction(uc, J, HEISENBERG_J, "J")
    bz      =   filled_bz(uc, kSize)
    H       =   Hamiltonian(uc, bz)
    DiagonalizeHamiltonian!(H)
    M       =   Model(uc, bz, H ; T = 1e-3, filling = (1 - δ) / 2, stat = -1)
    return TBMFTModel(M, vcat(onsite_orders(uc), bond_orders(uc)), [JParam], InterQuarticToHopping, Dict{String, Float64}("ij" => 1.0, "ii" => 0.0, "jj" => 0.0))
end

##### attractive Hubbard model: on-site s-wave pairing
function attractive_hubbard(U ; kSize = 15)
    hop, pair   =   new_cell(:square), new_cell(:square)
    add_nn_hopping!(hop, 1.0)
    UParam  =   Param(U, 4)
    AddIsotropicBonds!(UParam, hop, 0.0, HUBBARD_U, "U")
    bz      =   filled_bz(hop, kSize)
    H       =   Hamiltonian(hop, pair, bz)
    DiagonalizeHamiltonian!(H)
    M       =   BdGModel(hop, pair, bz, H ; T = 1e-3, filling = 0.4, stat = -1)
    return BdGMFTModel(M, onsite_orders(hop), onsite_orders(pair ; channels = PAIRING_CHANNELS[1:1]), [UParam], IntraQuarticToHopping, IntraQuarticToPairing)
end

##### renormalized square t-J model of models.jl (s-hopping + d-wave pairing), with J as a parameter
function SquaretJ_J(J ; filling = 0.4, kSize = 15)
    HoppingUC, PairingUC    =   UnitCell([[1.0, 1.0], [1.0, -1.0]], 2, 2), UnitCell([[1.0, 1.0], [1.0, -1.0]], 2, 2)
    for uc in (HoppingUC, PairingUC)
        AddBasisSite!(uc, [0.0, 0.0])
        AddBasisSite!(uc, [1.0, 0.0])
    end
    δ       =   1 - 2 * filling
    t1      =   Param(-2δ / (1 + δ), 2)
    AddIsotropicBonds!(t1, HoppingUC, 1.0, SPIN[4], "t1")
    JParam  =   Param(J * 4 / (1 + δ)^2, 4)
    AddIsotropicBonds!(JParam, HoppingUC, 1.0, HEISENBERG_J, "J")
    CreateUnitCell!(HoppingUC, [t1])
    bz      =   filled_bz(HoppingUC, kSize)
    t_s     =   Param(1.0, 2)
    AddIsotropicBonds!(t_s, HoppingUC, 1.0, SPIN[4], "s Hopping")
    p_d     =   Param(1.0, 2)
    for (offset, sign) in (([0, 0], 1), ([-1, -1], 1), ([0, -1], -1), ([-1, 0], -1))
        AddAnisotropicBond!(p_d, PairingUC, 1, 2, offset, sign * SPIN[2], 1.0, "d_x^2-y^2 Pairing")
    end
    H       =   Hamiltonian(HoppingUC, PairingUC, bz)
    DiagonalizeHamiltonian!(H)
    M       =   BdGModel(HoppingUC, PairingUC, bz, H ; T = 0.001, filling = filling, stat = -1)
    return BdGMFTModel(M, [t_s], [p_d], [JParam], InterQuarticToHopping, InterQuarticToPairing)
end

random_start(n ; seed = 3) = (rng = Xoshiro(seed); 0.3 .* (randn(rng, n) .+ im .* randn(rng, n)))

@testset "Mean-field energies" begin

    @testset "free fermions: E = Σ occupied band energies" begin
        for lattice in LATTICES
            mft     =   hubbard_mft(lattice ; U = 0.0, kSize = 15)
            SolveMFT!(mft, zeros(ComplexF64, norders(mft)) ; max_iter = 3, tol = 1e-12)
            M       =   mft.model
            nF(E)   =   1 / (exp((E - M.mu) / M.T) + 1)
            bands   =   sum(sum(E -> E * nF(E), b) for b in M.Ham.bands) / (prod(M.bz.gridSize) * length(M.uc.basis))
            @test mft.MFTEnergy[end] ≈ bands atol = 1e-8
        end
    end

    ##### T must be small: at finite T the identity holds for the free energy E - TS, not for E.

    @testset "Hellmann-Feynman: $(name)" for (name, make, λ, x0, restol) in (
            ("Hubbard, honeycomb Néel (on-site Hartree-Fock)",   U -> hubbard_mft(:honeycomb ; U = U, kSize = 15), 4.0, random_start(20), 1e-8),
            ("t-J, Fock channel only (inter-site Fock)",          tJ_fock,                                          1.5, random_start(24), 1e-8),
            ("t-J d-wave (inter-site pairing)",                  SquaretJ_J,                                       0.5, [0.2, 0.2],      1e-8),
            ("attractive Hubbard (on-site pairing)",             attractive_hubbard,                              -1.5, random_start(10), 1e-8))

        dEdλ, Hλ, residual  =   hellmann_feynman(make, λ, x0)
        @test residual < restol
        @test dEdλ ≈ Hλ rtol = 1e-3
    end

    ##### ------------------------------------------- finite temperature -------------------------------------------

    solve!(mft, x)  =   (sc = SolveMFT!(mft, copy(x) ; max_iter = 800, tol = 1e-11); (sc.VOuts[end], maximum(abs.(sc.VOuts[end] - sc.VIns[end]))))
    at_T(mft, T)    =   (mft.model.T = T; mft)

    @testset "free fermions: F = Ω + μN in closed form" begin
        for lattice in LATTICES
            mft     =   hubbard_mft(lattice ; U = 0.0, T = 0.2, kSize = 15)
            solve!(mft, zeros(ComplexF64, norders(mft)))
            M       =   mft.model
            Es      =   reduce(vcat, M.Ham.bands)
            Nsites  =   length(M.Ham.bands) * length(M.uc.basis)
            Ω       =   -M.T * sum(log1p.(exp.(-(Es .- M.mu) ./ M.T))) / Nsites
            N       =   sum(1 ./ (exp.((Es .- M.mu) ./ M.T) .+ 1)) / Nsites
            @test GetMFTFreeEnergy(mft) ≈ Ω + M.mu * N atol = 1e-10
            @test GetMFTEntropy(hubbard_mft(lattice ; U = 0.0, T = 0.0, kSize = 3)) == 0.0
        end
    end

    @testset "dF/dT = -S: $(name)" for (name, make, T, x0) in (
            ("Hubbard, honeycomb Néel",         T -> hubbard_mft(:honeycomb ; U = 4.0, T = T, kSize = 15), 0.2,  random_start(20)),
            ("t-J, Fock channel only",          T -> at_T(tJ_fock(1.5), T),                                0.05, random_start(24)),
            ("t-J d-wave (BdG)",                T -> at_T(SquaretJ_J(0.5), T),                             0.03, [0.2, 0.2]))

        mft         =   make(T)
        x, residual =   solve!(mft, x0)
        h           =   1e-3 * T
        Fp, Fm      =   make(T + h), make(T - h)
        solve!(Fp, x)
        solve!(Fm, x)
        @test residual < 1e-8
        @test (GetMFTFreeEnergy(Fp) - GetMFTFreeEnergy(Fm)) / (2h) ≈ -GetMFTEntropy(mft) rtol = 2e-3
    end

    @testset "finite-T Hellmann-Feynman holds for F, not E" begin
        ##### metallic t-J normal state with every channel on, at T = 0.05
        function tJ_metal(J)
            uc      =   new_cell(:square)
            add_nn_hopping!(uc, 0.2)
            JParam  =   nn_interaction(uc, J, HEISENBERG_J, "J")
            bz      =   filled_bz(uc, 15)
            H       =   Hamiltonian(uc, bz)
            DiagonalizeHamiltonian!(H)
            M       =   Model(uc, bz, H ; T = 0.05, filling = 0.4, stat = -1)
            return TBMFTModel(M, vcat(onsite_orders(uc), bond_orders(uc)), [JParam], InterQuarticToHopping)
        end
        λ, h        =   0.6, 6e-4
        mft         =   tJ_metal(λ)
        x, residual =   solve!(mft, random_start(24))
        plus, minus =   tJ_metal(λ + h), tJ_metal(λ - h)
        solve!(plus, x)
        solve!(minus, x)
        Hλ          =   interaction_energy(mft) / λ
        @test residual < 1e-8
        @test (GetMFTFreeEnergy(plus) - GetMFTFreeEnergy(minus)) / (2h) ≈ Hλ rtol = 1e-4
        @test !isapprox((GetMFTEnergy(plus) - GetMFTEnergy(minus)) / (2h), Hλ ; rtol = 1e-2)
    end
end
