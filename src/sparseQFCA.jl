#-------------------------------------------------------------------------------------------
#=
    Purpose:    Include all relevant files for running sparseQFCA
    Author:     Iman Ghadimi, Mojtaba Tefagh - Sharif University of Technology
    Date:       July 2023
=#
#-------------------------------------------------------------------------------------------

module sparseQFCA

    using COBREXA, SparseArrays, HiGHS, JuMP, LinearAlgebra, Distributed, SharedArrays, SparseArrays

    include("Pre_Processing/Pre_processing.jl")
    include("Pre_Processing/Solve.jl")
    include("ConsistencyChecking/TheNaiveApproach.jl")
    include("ConsistencyChecking/SwiftCC.jl")
    include("QFCA/distributedQFCA.jl")
    include("QFCA/SQFCA.jl")
    include("QuantomeRedNet/QuantomeReducer.jl")
    include("CompressedCOBRA/CompressedFBA.jl")

    using .Pre_processing, .Solve, .TheNaiveApproach, .SwiftCC, .DistributedQFCA, .SQFCA, .QuantomeReducer, .CompressedFBA

end

#-------------------------------------------------------------------------------------------
