##### Smoke tests for the shared example code in Sample/common.jl (used by the notebooks), so it does not rot.

@testset "Example models (Sample/common.jl)" begin

    ##### expected number of unrestricted order parameters (sites × 4 + bonds × 4, plus bonds × 4 pairing for t-J)
    counts  =   Dict(:square => (24, 40), :triangular => (48, 84), :honeycomb => (20, 32), :kagome => (36, 60))

    for lattice in LATTICES
        @testset "$(lattice)" begin
            uc  =   new_cell(lattice)
            @test all(b -> norm(bond_vector(uc, b)) ≈ 1.0, nn_bonds(uc))

            hubbard     =   hubbard_mft(lattice ; U = 4.0, kSize = 6)
            heisenberg  =   heisenberg_mft(lattice ; kSize = 6)
            tJ          =   tJ_mft(lattice ; δ = 0.1, kSize = 6)
            @test norders(hubbard) == norders(heisenberg) == counts[lattice][1]
            @test norders(tJ) == counts[lattice][2]

            for mft in (hubbard, heisenberg, tJ)
                sc  =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 5, tol = 1e-12)
                @test all(isfinite, sc.VOuts[end]) && isfinite(mft.MFTEnergy[end])
            end
        end
    end

    @testset "Heisenberg: classical ordered states" begin
        ##### Unrestricted spinon mean-field theory with the Hartree channel reproduces the classical ground state:
        ##### |S| = 1/2 on every site, one spinon per site, and E/site = (z/2) J S² cos θ.
        classical   =   Dict(:square => -0.5, :triangular => -0.375, :honeycomb => -0.375, :kagome => -0.25)
        for lattice in LATTICES
            mft     =   heisenberg_mft(lattice ; kSize = 6)
            sc      =   SolveMFT!(mft, initial_guess(mft, lattice) ; max_iter = 300, tol = 1e-9)
            moments =   site_moments(mft)
            @test maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-9
            @test mft.MFTEnergy[end] ≈ classical[lattice] atol = 1e-6
            @test all(n -> isapprox(n, 1.0, atol = 1e-6), moments.density)
            @test all(s -> isapprox(norm(s), 0.5, atol = 1e-6), moments.spin)
        end
    end

    @testset "spin convention: ⟨S⟩ = tr(Sᵀ G) with G_ab = ⟨f†_a f_b⟩" begin
        ##### one site in a field h·S^μ: the spin points opposite to the field
        for μ in 1:3
            uc  =   UnitCell([[1.0, 0.0], [0.0, 1.0]], 2, 2)
            AddBasisSite!(uc, [0.0, 0.0])
            field   =   Param(1.0, 2)
            AddAnisotropicBond!(field, uc, 1, 1, [0, 0], SPIN[μ], 0.0, "field")
            CreateUnitCell!(uc, [field])
            bz  =   BZ([3, 3])
            FillBZ!(bz, uc)
            H   =   Hamiltonian(uc, bz)
            DiagonalizeHamiltonian!(H)
            M   =   Model(uc, bz, H ; T = 1e-3, filling = 0.5, stat = -1)
            with_logger(NullLogger()) do
                SolveModel!(M)
            end
            S   =   spin_expectation(M.Gr[1, 1])
            @test S[μ] ≈ -0.5 atol = 1e-6
            @test norm(S) ≈ 0.5 atol = 1e-6
        end
    end
end

@testset "chemical potential found when every band lies above zero" begin
    ##### Paramagnetic honeycomb Hubbard model at U = 8t: the Hartree shift U n/2 = 4t lifts all bands above zero.
    ##### TightBindingToolkit's GetMu! searches 10 .* bandwidth = [10, 70] here and fills every level (density 2);
    ##### SolveFilling! must find μ ≈ 4 and keep one electron per site.
    mft     =   hubbard_mft(:honeycomb ; U = 8.0, T = 0.5, kSize = 9)
    x0      =   [startswith(label, "n on site") ? 0.5 + 0.0im : 0.0im for label in order_labels(mft)]
    SolveMFT!(mft, x0 ; max_iter = 200, tol = 1e-10)
    @test all(n -> isapprox(n, 1.0 ; atol = 1e-8), site_moments(mft).density)
    @test 0.0 < mft.model.mu < 8.0
end
