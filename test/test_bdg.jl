##### Regression pins produced by this package (BdG solver, tol = 1e-10). Simple mixing and Anderson agree to ~1e-10.
##### Before v2.3 this case stalled at a residual of ~3e-4: MFTIterator solved for μ only to mu_tol = 1e-3,
##### which made each iteration noisy at that level (and shifted the answer by ~3e-4).
##### v2.3 also corrected the energy (hopping and pairing bonds now counted with their Hermitian conjugate).

@testset "Renormalized t-J model, d-wave BdG" begin
    mft     =   SquaretJ()
    sc      =   SolveMFT!(mft, [0.2, 0.2]; max_iter = 300, tol = 1e-10)

    @test maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-10
    @test length(sc.VOuts) <= 20                    ##### Anderson (the default) needs ~9 iterations, simple mixing ~80
    @test mft.MFTEnergy[end] ≈ -0.5920467158371475 atol = 1e-8
    @test sc.VOuts[end]      ≈ [0.1956511853282087, 0.12500794625657302] atol = 1e-8

    ##### the chemical-potential tolerance can still be loosened through F_kwargs (e.g. for speed)
    mft     =   SquaretJ()
    sc      =   SelfCons(MFTIterator, AndersonMixing, [0.2, 0.2]; F_args = (mft,), F_kwargs = Dict{Symbol, Any}(:mu_tol => 1e-3, :filling_tol => 1e-6))
    FixedPoint!(sc; max_iter = 30, tol = 1e-10)
    @test maximum(abs.(sc.VOuts[end] - sc.VIns[end])) > 1e-8      ##### noisy F: cannot converge tightly
end
