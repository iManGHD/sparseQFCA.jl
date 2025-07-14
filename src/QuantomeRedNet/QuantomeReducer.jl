#-----------------------------------------------------------------------------------------------------------------------------------------------------
#=
    Purpose:    Metabolic Network Compression based on Quantitative Flux Coupling Analysis and the concept of lossless compression.
    Author:     Iman Ghadimi, Mojtaba Tefagh - Sharif University of Technology
    Date:       May 2023
=#
#-----------------------------------------------------------------------------------------------------------------------------------------------------

module QuantomeReducer

export quantomeReducer

using COBREXA, SparseArrays, JuMP, LinearAlgebra, Distributed, SharedArrays, JSON, JLD2

import CDDLib
import Clarabel

import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel: Model
import AbstractFBCModels.CanonicalModel: Reaction, Metabolite, Gene, Coupling

import JSONFBCModels: JSONFBCModel
import SBMLFBCModels: SBMLFBCModel

include("../Pre_Processing/Solve.jl")
using .Solve

include("../Pre_Processing/Pre_processing.jl")
using .Pre_processing

include("../ConsistencyChecking/SwiftCC.jl")
using .SwiftCC

include("../QFCA/distributedQFCA.jl")
using .DistributedQFCA

include("../ConsistencyChecking/TheNaiveApproach.jl")
using .TheNaiveApproach

"""
    quantomeReducer(model)

The function is designed to perform metabolic network compression by removing blocked reactions, merge all the fully coupled reactions,
remove the eligible reactions by the DCE-induced compressions.It extracts relevant data, separates reversible and irreversible reactions,
corrects reversibility, and removes zero rows from the stoichiometric matrix. It processes flux coupling analysis to identify reaction
clusters and reactions to be removed. The function constructs a compressed metabolic network matrix and performs distributed optimization
for DCE-induced compressions. Finally, it generates information about the compression process and returns the compressed metabolic network matrix.

# INPUTS

- `model`:                     A CanonicalModel that has been built using COBREXA's `load_model` function.
- `CompressedModelName`        A string, name of the metabolic network..

# OPTIONAL INPUTS

- `SolverName`:                Name of the solver(default: HiGHS).
- `OctuplePrecision`:          A flag(default: false) indicating whether octuple precision should be used when solving linear programs.
- `removing`:                  A flag controlling whether reactions should be filtered out during the coupling determination phase of network analysis.
- `Tolerance`:                 A small number that represents the level of error tolerance.
- `printLevel`:                Verbose level (default: 1). Mute all output with `printLevel = 0`.

# OUTPUTS

- `CompressedModelName`        Name of the compressed metabolic network.
- `A`                          A `n` x `ñ` matrix representing the coefficients betwee reactions of original networks and reactions of compressed network.
- `reduct-map`                 A dictionary to save the representatives of eliminations.

# EXAMPLES

- Full input/output example
```julia
julia> CompressedModelName, A, compression_map = quantomeReducer(model, ModelName)
```

See also: `dataOfModel()`, `reversibility()`, `homogenization()`, `distributedReversibility_Correction()`, `distributedQFCA()`

"""

function quantomeReducer(model, ModelName, SolverName::String="HiGHS", OctuplePrecision::Bool=false, removing::Bool=false, Tolerance::Float64=1e-9, printLevel::Int=1)

    ## Extracte relevant data from input model

    S, Metabolites, Reactions, Genes, m, n, n_genes, lb, ub, c_vector = dataOfModel(model, printLevel)

    c_vector_initial = copy(c_vector)

    # Find the index of the first occurrence where the element in c_vector is equal to 1.0:
    index_c = findfirst(x -> x == 1.0, c_vector)
    index_c_initial = copy(index_c)

    # Use the found index to retrieve the corresponding element from the Reactions array:
    Biomass = Reactions[index_c]

    row_S, col_S = size(S)

    # Create an array of reaction IDs:
    Reaction_Ids = collect(1:n)
    irreversible_reactions_id, reversible_reactions_id = reversibility(lb, Reaction_Ids, printLevel)

    ## Separate reactions into reversible and irreversible sets

    irreversible_reactions_id = sort(irreversible_reactions_id)
    reversible_reactions_id = sort(reversible_reactions_id)

    ## Create a new instance of the input model with homogenous bounds

    #blocked_index = find_blocked_reactions(model)

    ModelObject_CC = Model_CC(S, Metabolites, Reactions, Genes, m, n, lb, ub)
    model_CC_Constructor(ModelObject_CC , S, Metabolites, Reactions, Genes, row_S, col_S, lb, ub)
    blocked_index, ν  = swiftCC(ModelObject_CC, SolverName, false, Tolerance, printLevel)

    blocked_index_rev = blocked_index ∩ reversible_reactions_id

    # Convert to Vector{Int64}:
    blocked_index_rev = convert(Vector{Int64}, blocked_index_rev)

    ## Correct Reversibility

    # Create an empty dictionary to store reaction bounds:
    Dict_bounds = Dict()

    # Iterate through each reaction to populate the dictionary with lower and upper bounds:
    for i = 1:n
        Dict_bounds[Reactions[i]] = lb[i], ub[i]
    end

    # Create a new Model_Correction object with the current data:
    ModelObject_Correction = Model_Correction(S, Metabolites, Reactions, Genes, m, n, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    # Reconstruct the corrected model with updated parameters:
    model_Correction_Constructor(ModelObject_Correction , S, Metabolites, Reactions, Genes, row_S, col_S, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    # Apply distributedReversibility_Correction() to the model and update Reversibility, S and bounds:
    S, lb, ub, irreversible_reactions_id, reversible_reactions_id = distributedReversibility_Correction(ModelObject_Correction, blocked_index_rev, SolverName, false)

    irreversible_reactions_id = sort(irreversible_reactions_id)
    reversible_reactions_id = sort(reversible_reactions_id)

    # Get the dimensions of the updated stoichiometric matrix:
    row_S, col_S = size(S)

    ## Count the number of reactions in each set

    n_irr = length(irreversible_reactions_id)
    n_rev = length(reversible_reactions_id)

    # Update the dictionary with the new lower and upper bounds after correction:
    for i = 1:n
        Dict_bounds[Reactions[i]] = lb[i], ub[i]
    end

    ## Obtain blocked_index, fctable, Fc_Coefficients, and Dc_Coefficients

    ModelObject_QFCA = Model_QFCA(S, Metabolites, Reactions, Genes, row_S, col_S, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    model_QFCA_Constructor(ModelObject_QFCA , S, Metabolites, Reactions, Genes, row_S, col_S, lb, ub, irreversible_reactions_id, reversible_reactions_id)
    # Convert to Vector{Int64}:
    blocked_index = convert(Vector{Int64}, blocked_index)
    fctable, Fc_Coefficients, Dc_Coefficients = distributedQFCA(ModelObject_QFCA, blocked_index, SolverName, false)

    # Get the dimensions of fctable:
    row_fctable, col_fctable = size(fctable)

    # Remove blocked reactions from Reaction_Ids:
    Reaction_Ids_noBlocked = setdiff(Reaction_Ids, blocked_index)

    # Initialize arrays:
    A_rows_original = Array{Int64}([])
    A_cols_compressed = Array{Int64}([])

    # Make copies of Reaction_Ids for later use:
    A_rows_original = copy(Reaction_Ids)
    A_cols_compressed = copy(Reaction_Ids)

    ## FC

    # Initialize dictionaries and lists:
    FC = Dict()
    FC_Final = Dict()
    FC_Coef = Dict()
    remove_list_FC = []
    c = 1

    ## Iterate over fctable to identify and store FC coefficients

    # Iterating over a range starting from 1 and ending at col:
    for i = 1:col_fctable
        # Nested loop iterating over a range starting from i+1 and ending at col:
        for j = i+1:col_fctable
            # Checking conditions for equality and i not equal to j:
            if (fctable[i,j] == fctable[j,i] == 1.0) && (i != j)
                # Assigning tuple to FC_Coef[c]:
                FC_Coef[c] = Reaction_Ids_noBlocked[i], Reaction_Ids_noBlocked[j], Fc_Coefficients[i,j]
                # Assigning tuple to FC[c]:
                FC[c] = Reaction_Ids_noBlocked[i], Reaction_Ids_noBlocked[j]
                # Incrementing the counter c by 1:
                c = c + 1
            end
        end
    end

    ## Process FC dictionary to remove duplicates and create FC_Final dictionary

    s = 1
    # Iterating over keys of FC after sorting them:
    for key ∈ sort(collect(keys(FC)))
        # Checking conditions for key comparison:
        if (key >= 2) && (FC[key][1] == FC[key-1][1])
            # Initializing an empty list called temp_list:
            temp_list = []
            # Initializing an empty list called delete_list:
            delete_list = []
            # Looping from 1 to key-1:
            for i = 1 : key-1
                # Checking condition for equality of FC elements:
                if FC[key][1] .== FC[i][1]
                    # Appending FC[i][2] to temp_list:
                    append!(temp_list, FC[i][2])
                    # Appending i to delete_list:
                    append!(delete_list, i)
                end
            end
            # Appending FC[key][2] to temp_list:
            append!(temp_list, FC[key][2])
            # Appending temp_list to remove_list_FC:
            append!(remove_list_FC, temp_list)
            # Assigning tuple to FC_Final[s]:
            FC_Final[s] = FC[key][1], temp_list
            # Incrementing the counter s by 1:
            s = s + 1
            # Looping over delete_list:
            for i ∈ delete_list
                # Deleting element i from FC_Final:
                delete!(FC_Final, i)
            end
        else
            FC_Final[s] = FC[s]  # Assigning FC[s] to FC_Final[s]
            s = s + 1  # Incrementing the counter s by 1
        end
    end

    # Remove duplicate FC cluster members:
    remove_list_FC = unique(remove_list_FC)

    # Sort FC cluster members:
    remove_list_FC = sort(remove_list_FC)

    # Remove FC clusters that contain reactions in remove_list_FC from FC_Final dictionary:
    for key ∈ sort(collect(keys(FC_Final)))
        if FC_Final[key][1] ∈ remove_list_FC
            delete!(FC_Final, key)
        end
    end

    # Initialize arrays for FC representatives and cluster members:
    FC_representatives = Array{Int64}([])
    FC_cluster_members = Array{Int64}([])

    # Iterate through the items in the dictionary
    for (key, value) ∈ FC_Final
        # Check if the second element of the tuple is an integer
        if isa(value[2], Integer)
            # Convert the integer to a Vector{Any}
            FC_Final[key] = (value[1], Any[value[2]])
        end
    end

    ## Iterate over sorted keys of FC_Final dictionary

    for key ∈ sort(collect(keys(FC_Final)))
        # Append cluster members to FC_cluster_members array:
        append!(FC_cluster_members, FC_Final[key][2])

        # Append representative reaction to FC_representatives array:
        append!(FC_representatives, FC_Final[key][1])

        # Check if index_c is among the cluster members:
        if index_c in FC_Final[key][2]
            # If true, update index_c to the representative reaction:
            index_c = FC_Final[key][1]

            # Update Biomass variable with the reaction corresponding to index_c:
            Biomass = Reactions[index_c]
        end
    end

    FC_representatives = sort(FC_representatives)
    FC_cluster_members = sort(FC_cluster_members)

    ## Create a new dictionary FC_Clusters

    FC_Clusters = Dict()

    # Initialize counter variable:
    c = 1

    # Iterate over sorted keys of FC_Final dictionary:
    for key ∈ sort(collect(keys(FC_Final)))
        # Assign each cluster from FC_Final to FC_Clusters with a numeric key:
        FC_Clusters[c] = FC_Final[key]

        # Increment counter for each cluster:
        c += 1
    end

    ## DC

    # Initialize an empty array to store the IDs of reactions to be removed:
    remove_list_DC = Array{Int64}([])

    # Iterate over the rows:
    for i = 1:row_fctable
        # Iterate over the columns:
        for j = 1:col_fctable
            # Check if the value at position (i, j) in fctable is equal to 4.0:
            if (4.0 ∈ fctable[i, :]) # && (Reaction_Ids_noBlocked[i] ∉ representatives)
                # If the condition is true, append the corresponding Reaction ID to remove_list_DC:
                append!(remove_list_DC, Reaction_Ids_noBlocked[i])
                # Exit the inner loop as the reaction has been found and added to remove_list_DC:
                break
            end
        end
    end

    # Sort the 'blocked_index', 'FC_cluster_members', and 'remove_list_DC' arrays:
    blocked_index = sort(blocked_index)
    FC_cluster_members = sort(FC_cluster_members)
    remove_list_DC = sort(remove_list_DC)

    # Sort the 'Reaction_Ids_noBlocked', 'irreversible_reactions_id', and 'reversible_reactions_id' arrays:
    Reaction_Ids_noBlocked = sort(Reaction_Ids_noBlocked)
    irreversible_reactions_id = sort(irreversible_reactions_id)
    reversible_reactions_id = sort(reversible_reactions_id)

    #remove_list_DC = setdiff(remove_list_DC, FC_representatives)

    # Create the 'Eliminations' array by taking the union of 'blocked_index', 'remove_list_DC', and 'FC_cluster_members':
    Eliminations = union(blocked_index, remove_list_DC, FC_cluster_members)

    # Sort Eliminations:
    Eliminations = sort(Eliminations)

    # Update 'A_cols_compressed' by removing elements in 'Eliminations' from the range 1 to 'n' in 'A_cols_compressed':
    A_cols_compressed = A_cols_compressed[setdiff(range(1, n), Eliminations)]

    ## Matrix A

    # Create a shared array 'A' of size (A_rows_original, A_cols_compressed) with initial values set to false:
    n = length(A_rows_original)
    ñ = length(A_cols_compressed)

    A = SharedArray{Float64, 2}((n,ñ), init = false)

    ## Blocked

    # Iterate over indices from 1 to n:
    for i = 1:n
        if i ∈ blocked_index
            # Set the entire row to 0.0 in 'A' for the indices present in 'blocked_index':
            A[i, :] .= 0.0
        end
    end

    ## I

    # Iterate over indices from 1 to ñ:
    for i = 1:ñ
        # Set the corresponding element to 1.0 in 'A' for each index in 'A_cols_compressed':
        A[A_cols_compressed[i], i] = 1.0
    end

    # Create FC_Final_coefficients
    FC_Clusters_coefficients = Dict()
    c = 1
    for key in keys(sort(FC_Clusters))
        reaction_id, coupled_indices = FC_Clusters[key]
        reaction_id_original = findfirst(x -> x == reaction_id, Reaction_Ids_noBlocked)
        coefficients = Array{Float64}([])
        for idx in coupled_indices
            idx_original = findfirst(x -> x == idx, Reaction_Ids_noBlocked)
            coefficient = Fc_Coefficients[reaction_id_original, idx_original]
            append!(coefficients, coefficient)
        end
        FC_Clusters_coefficients[c] = (reaction_id, coupled_indices, coefficients)
        c += 1
    end

    ## FC

    for key ∈ sort(collect(keys(FC_Clusters_coefficients)))
        reaction_id, coupled_indices, coefficients = FC_Clusters_coefficients[key]
        if reaction_id ∈ A_cols_compressed
            col = findfirst(x -> x == reaction_id, A_cols_compressed)
            for (row, coefficient) in zip(coupled_indices, coefficients)
                A[row, col] = 1 / coefficient
            end
        end
    end

    ## DC

    DCE = Dict()
    counter = 1

    row, col = size(S)
    S, Metabolites_compressed, Metabolites_elimination  = remove_zeroRows(S, Metabolites)
    row, col = size(S)

    optimal_Number = 0
    infeasible_Number = 0
    almostoptimal_Number = 0

    # Check if we're using octuple precision (very high precision floating-point numbers):
    if OctuplePrecision
        # Define a model_irr using GenericModel from Clarabel.jl:
        model_local = GenericModel{BigFloat}(Clarabel.Optimizer{BigFloat})

        # Set verbose attribute to false (disable verbose output):
        set_attribute(model_local, "verbose", false)

        # Set absolute tolerance for gap convergence to 1e-32:
        set_attribute(model_local, "tol_gap_abs", 1e-32)

        # Set relative tolerance for gap convergence to 1e-32:
        set_attribute(model_local, "tol_gap_rel", 1e-32)
    else
        # If not using octuple precision, change the solver based on the solvername:
        model_local, solver = changeSparseQFCASolver(SolverName)
    end

    # Remove blocked indices from the stoichiometric matrix S
    S_noBlocked = S[:, setdiff(1:n, blocked_index)]
    n_noBlocked = size(S_noBlocked, 2)

    # Define the decision variables λ (for reactions), ν (for metabolites), and t (a scalar variable):
    @variable(model_local, λ[1:n_noBlocked])
    @variable(model_local, ν[1:m])
    @variable(model_local, t)

    @objective(model_local, Min, t)

    # Constraint 1: λ and t must satisfy the NormOneCone (a type of norm constraint):
    con1 = @constraint(model_local, [t; λ] in MOI.NormOneCone(1 + length(λ)))

    # Constraint 2: The dual variable λ must be equal to the transposed stoichiometric matrix (S') times ν:
    con2 = @constraint(model_local, λ == S_noBlocked' * ν)

    # Create a new array without the blocked indices
    lb_noBlocked = [lb[i] for i in 1:length(lb) if i ∉ blocked_index]
    irreversible_reactions_id_noBlocked, reversible_reactions_id_noBlocked = reversibility(lb_noBlocked, Reaction_Ids_noBlocked, 0)

    for i in reversible_reactions_id_noBlocked
        index = findfirst(x -> x == i, Reaction_Ids_noBlocked)
        # Constraint 3: Ensure that λ is 0 for reversible reactions (the reaction flux is zero for reversible reactions):
        con3 = @constraint(model_local, λ[index] == 0.0)
    end

    Eliminations = setdiff(Eliminations, blocked_index)

    Eliminations_noBlocked = Array{Int64}([])
    for i in Eliminations
        index = findfirst(x -> x == i, Reaction_Ids_noBlocked)
        append!(Eliminations_noBlocked, index)
    end

    remove_list_DC_noBlocked = Array{Int64}([])
    for i in remove_list_DC
        index = findfirst(x -> x == i, Reaction_Ids_noBlocked)
        append!(remove_list_DC_noBlocked, index)
    end

    ## Perform distributed optimization for each reaction in the remove_list_DC list
    counter = 1

    for i ∈ remove_list_DC_noBlocked

        # Save i as the row number to process the current reaction:
        row = remove_list_DC[counter]

        counter += 1

        if row ∉ FC_cluster_members

            # Compute the linear combination for DCE by excluding the current reaction (i):
            DCE_LinearCombination = setdiff(1:n_noBlocked, i)

            ## Define additional constraints for the optimization problem

            # Constraint 4: Set λ to zero for specific reactions in the Eliminations ∩ DCE_LinearCombination set:
            con4 = @constraint(model_local, [j in Eliminations_noBlocked ∩ DCE_LinearCombination], λ[j] == 0.0)

            # Constraint 5: Set λ to be non-negative for the remaining reactions in DCE_LinearCombination:
            con5 = @constraint(model_local, [j in DCE_LinearCombination], λ[j] >= 0.0)

            # Constraint 6: Force λ for the current reaction (i) to be -1:
            con6 = @constraint(model_local, -1.0 - Tolerance <= λ[i] <= -1.0 + Tolerance)

            # Solve the optimization problem for the current setup:
            optimize!(model_local)

            # Check the optimization result
            status = termination_status(model_local)

            # Check the optimization result
            status = termination_status(model_local)
            if status == MOI.OPTIMAL
                optimal_Number += 1

            elseif status == MOI.INFEASIBLE
                infeasible_Number += 1

            elseif status == MOI.ALMOST_OPTIMAL
                almostoptimal_Number += 1

            #elseif status == MOI.UNBOUNDED
                #println("Model is unbounded.")
            else
                println("Optimization was stopped with status ", status)
            end

            # Get the values of the λ variables after solving the optimization problem:
            λ_vec = value.(λ)

            # Create an empty array to store the indices of the non-zero λ values:
            cols = Array{Int64}([])

            DCE_list = Array{Int64}([])

            # Loop through the λ vector and check for values above the defined tolerance:
            for i = 1:length(λ_vec)
                if λ_vec[i] > 1e-6
                    # Find the column index of the corresponding reaction in A_cols_compressed:
                    index = Reaction_Ids_noBlocked[i]
                    append!(DCE_list, index)
                    index_final = findfirst(x -> x == index, A_cols_compressed)
                    # Update the corresponding value in matrix A for the current row:
                    # Assign your value
                    if (status == MOI.OPTIMAL)
                        A[row, index_final] = λ_vec[i]
                    end

                    if (status == MOI.ALMOST_OPTIMAL)
                        A[row, index_final] = λ_vec[i]
                    end
                end
            end

            ## Condition 4: Remove all constraints in con4 from the model once used

            constraint_refs_con4 = [con4[i] for i in eachindex(con4)]
            for i ∈ constraint_refs_con4
                delete(model_local, i)  # Remove the constraint from the model
                unregister(model_local, :i)  # Unregister the constraint for clean-up
            end

            ## Condition 5: Remove all constraints in con5 from the model once used

            constraint_refs_con5 = [con5[i] for i in eachindex(con5)]
            for i ∈ constraint_refs_con5
                delete(model_local, i)  # Remove the constraint from the model
                unregister(model_local, :i)  # Unregister the constraint for clean-up
            end

            ## Condition 6: Remove the current λ constraint (con6) from the model

            delete(model_local, con6)  # Remove the specific λ[i] == -1 constraint
            unregister(model_local, :con6)  # Unregister this constraint for clean-up

            #printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)
        end
    end

    for key ∈ sort(collect(keys(FC_Clusters_coefficients)))
        if FC_Clusters_coefficients[key][1] in remove_list_DC
            for (row, coefficient) in zip(FC_Clusters_coefficients[key][2], FC_Clusters_coefficients[key][3])
                A[row, :] = (1 / coefficient) .* A[FC_Clusters_coefficients[key][1], :]
            end
        end
    end

    # Get the number of rows and columns in matrix A:
    row_A, col_A = size(A)

    # Get the number of rows and columns in the stoichiometric matrix S:
    row_S, col_S = size(S)

    A_sparse = sparse(Array(A))
    dropzeros!(A_sparse)

    # Compute the modified stoichiometric matrix S̃ by multiplying S with A:
    S̃ = S * A

    S̃ = sparse(Array(S̃))
    dropzeros!(S̃)

    # Remove any zero rows from the matrix S̃ and get the compressed metabolite list:
    S̃, Metabolites_compressed, Metabolites_elimination = remove_zeroRows(S̃, Metabolites)

    # Get the dimensions of the compressed stoichiometric matrix S̃:
    row_S̃, col_S̃ = size(S̃)

    # Identify the reactions that have been eliminated (not part of A_cols_compressed):
    Reactions_elimination = Reactions[setdiff(range(1, n), A_cols_compressed)]

    # Get the compressed list of reactions corresponding to the columns in A_cols_compressed:
    R̃ = Reactions[A_cols_compressed]

    # Make a copy of the compressed metabolites to M̃:
    M̃ = copy(Metabolites_compressed)

    ## Compression Map

    # Creating an empty dictionary called compression_map:
    compression_map = Dict()

    # Initializing the counter c to 1:
    c = 1

    # Iterating over each column of matrix A:
    for col ∈ eachcol(A)
        # Converting the current column to a sparse vector:
        col = sparsevec(col)
        # Finding the non-zero indices and values in the sparse vector:
        nonzero_indices, nonzero_values = findnz(col)
        index = findfirst(x -> x == R̃[c], Reactions)
        # Assigning the tuple (R̃[c], nonzero_indices) to the key c in compression_map:
        compression_map[c] = index, nonzero_indices
        # Incrementing the counter c by 1:
        c += 1
    end

    ## Update lb & Up

    # Iterate through each reaction in the model to update the lower and upper bounds:
    for i ∈ model.reactions
        # Set the lower bound for the reaction using the pre-calculated Dict_bounds:
        i.second.lower_bound = Dict_bounds[i.first][1]
        # Set the upper bound for the reaction using the pre-calculated Dict_bounds:
        i.second.upper_bound = Dict_bounds[i.first][2]
    end

    # Filter out reactions that are in the list of eliminated reactions:
    filter!(pair -> !(pair.first in Reactions_elimination), model.reactions)

    # Filter out metabolites that are in the list of eliminated metabolites:
    filter!(pair -> !(pair.first in Metabolites_elimination), model.metabolites)

    ## Update Stoichiometry Matrix

    # Iterate through each reaction in the model to clear the stoichiometry:
    for i ∈ model.reactions
        # Collect the keys from the stoichiometry map of the current reaction:
        for key ∈ collect(keys(i.second.stoichiometry))
            # Remove the stoichiometry entry for the current key:
            delete!(i.second.stoichiometry, key)
        end
    end

    # Iterate through each reaction to update the stoichiometry based on the modified matrix:
    for i ∈ model.reactions
        # Find the index of the current reaction in the compressed reaction list R̃:
        index_col = findfirst(x -> x == i.first, R̃)

        # Get the corresponding column in the modified stoichiometric matrix S̃:
        stoichiometry_vector = S̃[:,index_col]

        # Initialize a counter for metabolites:
        met = 1

        # Loop through the stoichiometry vector to update the stoichiometry for each metabolite:
        for c = 1:length(stoichiometry_vector)
            # Push the metabolite and its corresponding stoichiometry value to the reaction:
            push!(i.second.stoichiometry, "$(M̃[met])" => stoichiometry_vector[c])
            met += 1
        end
    end

    ## Update Biomass

    # Iterate through the compression map to check and update the Biomass reaction:
    for (key, value) ∈ sort(compression_map)
        # If the biomass reaction (index_c) is in the compression map, update its index:
        if index_c ∈ compression_map[key][2]
            index_c = compression_map[key][1]
            Biomass = Reactions[index_c]
        end
    end

    ## Update Objective coefficient

    # Iterate through the reactions to set the objective coefficient for the biomass reaction:
    for i ∈ model.reactions
        # If the current reaction matches the biomass reaction, set its objective coefficient to 1.0:
        if i.first == Biomass
            i.second.objective_coefficient = 1.0
        end
    end

    S_compressed, Metabolites_compressed, Reactions_compressed, Genes_compressed, m_compressed, n_compressed, n_genes_compressed, lb_compressed, ub_compressed, c_vector_compressed = dataOfModel(model, 0)

    display(S_compressed)

    dropzeros!(S_compressed)

    display(S_compressed)

    for i ∈ model.reactions
        index_Reactions_compressed = findfirst(x -> x == i.first, Reactions_compressed)
        i.second.objective_coefficient = c_vector_compressed[index_Reactions_compressed]
    end

    CompressedModelName = ModelName * "_compressed"
    model_compressed_json = convert(JSONFBCModel, model)
    save_model(model_compressed_json, "../src/QuantomeRedNet/CompressionResults/$CompressedModelName.json")

    # Read the JSON file
    data = JSON.parsefile("../src/QuantomeRedNet/CompressionResults/$CompressedModelName.json")

    # Process reactions
    if haskey(data, "reactions")
        for reaction in data["reactions"]
            if haskey(reaction, "gene_reaction_rule") && !isempty(reaction["gene_reaction_rule"])
                rule = reaction["gene_reaction_rule"]

                # Replace "()" with an empty string
                if rule == "()"
                    reaction["gene_reaction_rule"] = ""
                else
                    # Replace "&&" with "and" and "||" with "or"
                    reaction["gene_reaction_rule"] = replace(rule, "&&" => "and", "||" => "or")
                end
            end
        end
    end

    # Write the corrected JSON file
    open("../src/QuantomeRedNet/CompressionResults/$CompressedModelName.json", "w") do file
        JSON.print(file, data, 1)  # Use 'indent=1' for indentation
    end

    println("typeof A = $(typeof(A))")

    A_sparse = sparse(Array(A))
    dropzeros!(A_sparse)

    println("typeof A = $(typeof(A_sparse))")

    # Save matrix to a file
    @save "../src/QuantomeRedNet/CompressionResults/A_$CompressedModelName.jld2" A_sparse

    ## Print out results if requested

    if printLevel > 0
        printstyled("Metabolic Network compressions:\n"; color=:cyan)
        if OctuplePrecision
            printstyled("Solver = Clarabel \n"; color=:green)
        else
            printstyled("Solver = $SolverName\n"; color=:green)
        end
        printstyled("Tolerance = $Tolerance\n"; color=:magenta)
        println("Original Network:")
        println("S           : $(row_S) x $(col_S)")
        println("Genes       : $(length(Genes))")
        println("Metabolites : $(m)")
        println("Reactions   : $(n)")
        println("compressed Network:")
        println("S           : $(row_S̃) x $(col_S̃)")
        println("Genes       : $(length(Genes_compressed))")
        println("Metabolites : $(length(Metabolites_compressed))")
        println("Reactions   : $(length(R̃))")
        println("A matrix    : $(row_A) x $(col_A)")
    end
    return CompressedModelName, A, compression_map
end

end

#-----------------------------------------------------------------------------------------------------------------------------------------------------
