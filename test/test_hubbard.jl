##### Reference values are regression pins produced by this package (TB solver, tol = 1e-10),
##### not independent results: a change here means the physics output changed and needs to be understood.
##### v2.3: energies changed because GetMFTEnergy now counts inter-site bonds with their Hermitian conjugate (checked in test_energies.jl);
##### the order parameters did not change.

@testset "Square-lattice Hubbard, U = 8t, half filling" begin
    fm      =   SquareHubbard()
    scFM    =   SolveMFT!(fm, [0.1, 0.5, 0.5]; max_iter = 300, tol = 1e-10)
    afm     =   SquareHubbard()
    scAFM   =   SolveMFT!(afm, [0.1, 0.99, -0.99]; max_iter = 300, tol = 1e-10)

    @test maximum(abs.(scFM.VOuts[end]  - scFM.VIns[end]))  < 1e-8
    @test maximum(abs.(scAFM.VOuts[end] - scAFM.VIns[end])) < 1e-8
    @test length(scAFM.VOuts) <= 15                 ##### Anderson (the default) needs ~7 iterations, simple mixing ~30

    @test fm.MFTEnergy[end]  ≈ -2.000039506172839 atol = 1e-8
    @test scFM.VOuts[end]    ≈ [0.002222222222222222, 0.9955555555555555, 0.9955555555555555] atol = 1e-8
    @test afm.MFTEnergy[end] ≈ -2.465877982708125 atol = 1e-8
    @test scAFM.VOuts[end]   ≈ [0.10898434707592, 0.89274946264358, -0.89274946264358] atol = 1e-8

    ##### the Néel state beats the ferromagnet at half filling, with equal and opposite moments
    @test afm.MFTEnergy[end] < fm.MFTEnergy[end]
    @test scAFM.VOuts[end][2] ≈ -scAFM.VOuts[end][3]
end

@testset "Square-lattice Hubbard ferromagnet: two self-consistent solutions" begin
    ##### From the same start, simple mixing stops at a metastable, less polarized FM fixed point;
    ##### Anderson (the default since v2.3) reaches the lower-energy one pinned above. Both are genuine fixed points.
    fm      =   SquareHubbard()
    sc      =   SolveMFT!(fm, [0.1, 0.5, 0.5]; Update = SimpleMixing, max_iter = 300, tol = 1e-10)

    @test maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8
    @test fm.MFTEnergy[end]  ≈ -1.9994337059287257 atol = 1e-8
    @test sc.VOuts[end]      ≈ [0.01091686756208, 0.97777777777778, 0.97777777777778] atol = 1e-8
    @test fm.MFTEnergy[end]  > -2.000039506172839
end

@testset "1D Hubbard chain, U = 6t (1D Green's function in DecomposeGr)" begin
    mft     =   ChainHubbard()
    @test ndims(mft.model.Gr) == 1
    sc      =   SolveMFT!(mft, [0.1, 0.9, -0.9]; max_iter = 300, tol = 1e-10)

    @test maximum(abs.(sc.VOuts[end] - sc.VIns[end])) < 1e-8
    @test mft.MFTEnergy[end] ≈ -1.8240953763634813 atol = 1e-8
    @test sc.VOuts[end]      ≈ [0.15744988190597806, 0.8922988470759371, -0.8922988470759371] atol = 1e-8
end

@testset "Inter/Intra decomposition guard" begin
    mft     =   SquareHubbard(kSize = 3)
    M, orders, interactions     =   mft.model, mft.HoppingOrders, mft.Interactions

    ##### on-site Hubbard with an inter-site decomposition
    @test_throws ArgumentError TBMFTModel(M, orders, interactions, InterQuarticToHopping)
    @test_throws ArgumentError TBMFTModel(M, orders, interactions, Function[InterQuarticToHopping], mft.MFTScaling)

    ##### nearest-neighbour interaction with an on-site decomposition
    VParam  =   Param(1.0, 4)
    AddIsotropicBonds!(VParam, M.uc, 1.0, DensityToPartonCoupling(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2)), "V")
    @test_throws ArgumentError TBMFTModel(M, orders, [VParam], IntraQuarticToHopping)

    ##### correct pairings, and user-defined decompositions, are accepted
    @test TBMFTModel(M, orders, interactions, IntraQuarticToHopping) isa TBMFTModel
    @test TBMFTModel(M, orders, [VParam], InterQuarticToHopping) isa TBMFTModel
    @test TBMFTModel(M, orders, interactions, (U, Chis) -> Dict("ii" => zeros(ComplexF64, 2, 2))) isa TBMFTModel
end
