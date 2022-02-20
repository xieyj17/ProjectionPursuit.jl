@testset "gensphere" begin
    unit_sphere = gensphere(10,3)
    @test size(unit_sphere) == (10,3)
end