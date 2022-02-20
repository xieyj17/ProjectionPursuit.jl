@testset "gensphere" begin
    unit_sphere = gensphere(10,4)
    @test size(unit_sphere) == (10,4)
end