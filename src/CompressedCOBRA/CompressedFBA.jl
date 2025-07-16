#-----------------------------------------------------------------------------------------------------------------------------------------------------
#=
    Purpose:     Compressed Flux Balance Analysis (FBA) implementation.
    Author:      Iman Ghadimi Deylami, Mojtaba Tefagh - Sharif University of Technology
    Date:        July 2025
=#
#-----------------------------------------------------------------------------------------------------------------------------------------------------

module CompressedFBA

export FBA, correctedFBA, compressedFBA

using JuMP, HiGHS, SparseArrays, LinearAlgebra

include("../Pre_Processing/Solve.jl")
using .Solve

include("../Pre_Processing/Pre_processing.jl")
using .Pre_processing

include("../ConsistencyChecking/SwiftCC.jl")
using .SwiftCC

"""
    FBA(model, modelName)

The function performs Flux Balance Analysis (FBA) to optimize the flux distribution in a metabolic network. FBA aims to
maximize or minimize an objective function, such as biomass production, subject to stoichiometric and capacity constraints.

# INPUTS

- `model`:                     A CanonicalModel that has been built using COBREXA's `load_model` function.
- `modelName`:                 A string, name of the metabolic network..

# OPTIONAL INPUTS

- `printLevel`:                Verbose level (default: 1). Mute all output with `printLevel = 0`.

# OUTPUTS

- `V_initial`                  Flux vector as `Vector{Float64}`..
- `objective_value`            Objective function value.

# EXAMPLES

- Full input/output example
```julia
julia> V_initial, objective_value = FBA(model, "e_coli_core")
```

See also: `dataOfModel()`, `changeSparseQFCASolver()`

"""

function FBA(model, modelName, printLevel::Int=1)

    if printLevel > 0
        printstyled("FBA:\n"; color=:yellow)
    end

    ## Extract model data

    S, Metabolites, Reactions, Genes, m, n, n_genes, lb, ub, c_vector = dataOfModel(model, 0)
    row_S, col_S = size(S)

    ## Define and set up the JuMP optimization model

    # Define the model:
    FBA_model, solver = changeSparseQFCASolver("HiGHS")
    # Add decision variables:
    @variable(FBA_model, lb[i] <= x[i = 1:n] <= ub[i])
    # Set the objective function:
    @objective(FBA_model, Max, (c_vector)'* x)
    @constraint(FBA_model, (S) * x .== 0)
    # Solve the model:
    optimize!(FBA_model)
    # Retrieve fluxes:
    V_initial = [value(x[i]) for i in 1:n]
    index_c = findfirst(x -> x == 1.0, c_vector)
    objective_value = JuMP.objective_value(FBA_model)

    ## Print out results if requested

    if printLevel > 0
        printstyled("Flux Balance Analysis - $modelName:\n"; color=:cyan)
        println("Metabolic Network:")
        println("S           : $(row_S) x $(col_S)")
        println("Genes       : $(length(Genes))")
        println("Metabolites : $(m)")
        println("Reactions   : $(n)")
        println("termination_status = $(termination_status(FBA_model))")
        println("objective_value = $objective_value")
        println("Biomass = $(Reactions[index_c]), Flux = $(V_initial[index_c])")
    end
    return V_initial, objective_value
end

"""
    correctedFBA(model, modelName)

The function performs Flux Balance Analysis (FBA) to optimize the flux distribution in a reversibility corrected metabolic network.
FBA aims to maximize or minimize an objective function, such as biomass production, subject to stoichiometric and capacity constraints.

# INPUTS

- `model`:                     A CanonicalModel that has been built using COBREXA's `load_model` function.
- `modelName`                  A string, name of the metabolic network.

# OPTIONAL INPUTS

- `printLevel`:                Verbose level (default: 1). Mute all output with `printLevel = 0`.

# OUTPUTS

- `V_correction`               Flux vector after correction as `Vector{Float64}`.
- `objective_value`            Corrected objective function value.

# EXAMPLES

- Full input/output example
```julia
julia> V_correction, objective_value = FBA(model, "e_coli_core")
```

See also: `dataOfModel()`, `reversibility()`, `swiftCC()`, `distributedReversibility_Correction()`, `changeSparseQFCASolver()`

"""

function correctedFBA(model, modelName, printLevel::Int=1)

    if printLevel > 0
        printstyled("correctedFBA:\n"; color=:yellow)
    end

    ## Extract model data

    S, Metabolites, Reactions, Genes, m, n, n_genes, lb, ub, c_vector = dataOfModel(model, 0)
    row_S, col_S = size(S)

    ## Identify reversible and irreversible reactions

    # Create an array of reaction IDs:
    Reaction_Ids = collect(1:n)
    # Separate reactions into reversible and irreversible sets:
    irreversible_reactions_id, reversible_reactions_id = reversibility(lb, Reaction_Ids)

    ## Detect blocked reactions

    # Create a new instance of the input model with homogenous bounds:
    ModelObject_CC = Model_CC(S, Metabolites, Reactions, Genes, m, n, lb, ub)
    blocked_index, dualVar = swiftCC(ModelObject_CC)

    blocked_index_rev = blocked_index ∩ reversible_reactions_id
    # Convert to Vector{Int64}:
    blocked_index_rev = convert(Vector{Int64}, blocked_index_rev)

    ## Correct the reversible reactions using distributed reversibility checking

    # Correct Reversibility:
    ModelObject_Correction = Model_Correction(S, Metabolites, Reactions, Genes, m, n, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    S, lb, ub, irreversible_reactions_id, reversible_reactions_id = distributedReversibility_Correction(ModelObject_Correction, blocked_index_rev)

    # Define the model
    FBA_model_correction, solver = changeSparseQFCASolver("HiGHS")
    # Add decision variables
    @variable(FBA_model_correction, lb[i] <= x[i = 1:n] <= ub[i])
    # Set the objective function
    @objective(FBA_model_correction, Max, (c_vector)'* x)
    @constraint(FBA_model_correction, (S) * x .== 0)
    # Solve the model
    optimize!(FBA_model_correction)

    V_correction = [value(x[i]) for i in 1:n]
    index_c = findfirst(x -> x == 1.0, c_vector)
    objective_value = JuMP.objective_value(FBA_model_correction)

    ## Print out results if requested

    if printLevel > 0
        printstyled("Corrected Flux Balance Analysis - $modelName:\n"; color=:cyan)
        println("Metabolic Network:")
        println("S           : $(row_S) x $(col_S)")
        println("Genes       : $(length(Genes))")
        println("Metabolites : $(m)")
        println("Reactions   : $(n)")
        println("termination_status = $(termination_status(FBA_model_correction))")
        println("objective_value = $objective_value")
        println("Biomass = $(Reactions[index_c]), Flux = $(V_correction[index_c])")

    end
    return V_correction, objective_value
end

"""
    compressedFBA(model, compressedModel, A, modelName)

The function performs Flux Balance Analysis (FBA) to optimize the flux distribution in a compressed metabolic network. FBA aims
to maximize or minimize an objective function, such as biomass production, subject to stoichiometric and capacity constraints.

# INPUTS

- `model`:                     A CanonicalModel that has been built using COBREXA's `load_model` function.
- `compressedmodel`:           A compressed CanonicalModel that has been built using COBREXA's `load_model` function.
- `A`:                         Transformation matrix (mapping compressed reactions to full model).
- `modelName`                  A string, name of the metabolic network.

# OPTIONAL INPUTS

- `printLevel`:                Verbose level (default: 1). Mute all output with `printLevel = 0`.

# OUTPUTS

- `V`                          Flux vector in the original network.
- `V_compressed`               Flux vector in the compressed network..
- `objective_value`            Objective value in compressed model.

# EXAMPLES

- Full input/output example
```julia
julia> V, V_compressed, objective_value = compressedFBA(model, compressedModel, A, "e_coli_core")
```

See also: `dataOfModel()`, `reversibility()`, `swiftCC()`, `distributedReversibility_Correction()`, `changeSparseQFCASolver()`, `remove_zeroRows()`

"""

function compressedFBA(model, compressedModel, A, modelName, printLevel::Int=1)

    if printLevel > 0
        printstyled("compressedFBA:\n"; color=:yellow)
    end

    ## Extract original model info

    S, Metabolites, Reactions, Genes, m, n, n_genes, lb, ub, c_vector = dataOfModel(model, 0)
    row_S, col_S = size(S)
    M = 1000.0
    representatives = [i for i in 1:n if lb[i] ∉ (0.0, -M, M) || ub[i] ∉ (0.0, -M, M)]
    index_c = findfirst(x -> x == 1.0, c_vector)

    ## Correct model reversibility

    # Create an array of reaction IDs:
    Reaction_Ids = collect(1:n)
    # Separate reactions into reversible and irreversible sets:
    irreversible_reactions_id, reversible_reactions_id = reversibility(lb, Reaction_Ids)
    # Create a new instance of the input model with homogenous bounds:
    ModelObject_CC = Model_CC(S, Metabolites, Reactions, Genes, m, n, lb, ub)
    blocked_index, dualVar = swiftCC(ModelObject_CC)

    blocked_index_rev = blocked_index ∩ reversible_reactions_id
    # Convert to Vector{Int64}:
    blocked_index_rev = convert(Vector{Int64}, blocked_index_rev)
    # Correct Reversibility:
    ModelObject_Correction = Model_Correction(S, Metabolites, Reactions, Genes, m, n, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    S, lb, ub, irreversible_reactions_id, reversible_reactions_id = distributedReversibility_Correction(ModelObject_Correction, blocked_index_rev)

    ## compressedModel

    S_compressed, Metabolites_compressed, Reactions_compressed, Genes_compressed, m_compressed, n_compressed, n_genes_compressed, lb_compressed, ub_compressed, c_vector_compressed = dataOfModel(compressedModel)
    row_S_compressed, col_S_compressed = size(S_compressed)

    S_compressed = dropzeros!(S_compressed)
    Reaction_Ids_compressed = collect(1:n_compressed)
    irreversible_reactions_id_compressed, reversible_reactions_id_compressed = reversibility(lb_compressed, Reaction_Ids_compressed, 0)

    # Find the index of the first occurrence where the element in c_vector is equal to 1.0 in Reduced Network:
    index_c_compressed = findfirst(x -> x == 1.0, c_vector_compressed)
    c_vector_compressed = A' * c_vector

    S_compressed, Metabolites_compressed, Metabolites_elimination = remove_zeroRows(S_compressed, Metabolites_compressed)

    # Define the model
    FBA_model_compressed, solver = changeSparseQFCASolver("HiGHS")

    # Decision variables
    @variable(FBA_model_compressed, x[1:n_compressed])

    t = 0.001

    # Constraint 1: Bounds for representative reactions
    for i in representatives
        @constraint(FBA_model_compressed, lb[i] <= dot(A[i, :], x))  # Lower bound constraint
        @constraint(FBA_model_compressed, dot(A[i, :], x) <= ub[i])  # Upper bound constraint
    end

    # Constraint 2: Non-negativity for irreversible reactions
    @constraint(FBA_model_compressed, [j in irreversible_reactions_id_compressed], x[j] >= 0.0)

    # Replace Norm-2 with Norm-Infinity constraints
    @constraint(FBA_model_compressed, S_compressed * x .<= t)
    @constraint(FBA_model_compressed, S_compressed * x .>= -t)

    # Set objective
    @objective(FBA_model_compressed, Max, dot(c_vector_compressed, x))

    # Solve the model
    optimize!(FBA_model_compressed)

    # Map results back
    V_compressed = [value(x[i]) for i in 1:n_compressed]
    V = A * V_compressed
    objective_value = JuMP.objective_value(FBA_model_compressed)

    ## Print out results if requested

    if printLevel > 0
        printstyled("Compressed Flux Balance Analysis - $modelName:\n"; color=:cyan)
        println("Original Metabolic Network:")
        println("S           : $(row_S) x $(col_S)")
        println("Genes       : $(length(Genes))")
        println("Metabolites : $(m)")
        println("Reactions   : $(n)")
        println("Compressed Metabolic Network:")
        println("S           : $(row_S_compressed) x $(col_S_compressed)")
        println("Genes       : $(n_genes_compressed)")
        println("Metabolites : $(m_compressed)")
        println("Reactions   : $(n_compressed)")
        println("termination_status = $(termination_status(FBA_model_compressed))")
        println("objective_value = $objective_value")
        println("Biomass = $(Reactions[index_c]), Flux = $(V[index_c])")
    end
    return V, V_compressed, objective_value
end

end

#-----------------------------------------------------------------------------------------------------------------------------------------------------
