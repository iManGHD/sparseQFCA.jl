
cd(@__DIR__)

# Add worker processes to the Julia distributed computing environment:

using Distributed

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:magenta)

addprocs(7)
println("Number of Proccess : $(nprocs())")
println("Number of Workers  : $(nworkers())")

### Import Libraries

using COBREXA, JuMP, Test, Distributed, JuMP, HiGHS, Clarabel, JSON, SparseArrays, LinearAlgebra, SharedArrays, JLD2, HDF5

# Include the necessary Julia files:
include("TestData.jl")
@everywhere include("../src/sparseQFCA.jl")

# Import required Julia modules:

using .TestData, .sparseQFCA

import AbstractFBCModels as A
import AbstractFBCModels.CanonicalModel: Model
import AbstractFBCModels.CanonicalModel: Reaction, Metabolite, Gene, Coupling
import JSONFBCModels: JSONFBCModel

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:magenta)

### sparseQFCA:

# Print a message indicating that sparseQFCA is being run on e_coli_core:
printstyled("sparseQFCA :\n"; color=:magenta)
printstyled("iIS312 :\n"; color=:yellow)

## Extracte relevant data from input model

S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, n_genes_iIS312, lb_iIS312, ub_iIS312, c_vector = sparseQFCA.dataOfModel(myModel_iIS312)
# Ensure that the bounds of all reactions are homogenous:
lb_iIS312, ub_iIS312 = sparseQFCA.homogenization(lb_iIS312, ub_iIS312)
# Separate reactions into reversible and irreversible sets:
# Create an array of reaction IDs:
Reaction_Ids_iIS312 = collect(1:n_iIS312)
irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312 = sparseQFCA.reversibility(lb_iIS312, Reaction_Ids_iIS312)
# Create a new instance of the input model with homogenous bounds:
ModelObject_CC_iIS312 = sparseQFCA.Model_CC(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, lb_iIS312, ub_iIS312)
blocked_index_iIS312, dualVar_iIS312 = sparseQFCA.swiftCC(ModelObject_CC_iIS312)
blocked_index_rev_iIS312 = blocked_index_iIS312 ∩ reversible_reactions_id_iIS312
# Convert to Vector{Int64}
blocked_index_rev_iIS312 = convert(Vector{Int64}, blocked_index_rev_iIS312)
# Correct Reversibility:
ModelObject_Crrection_iIS312 = sparseQFCA.Model_Correction(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, lb_iIS312, ub_iIS312, irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312)
S_iIS312, lb_iIS312, ub_iIS312, irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312 = sparseQFCA.distributedReversibility_Correction(ModelObject_Crrection_iIS312, blocked_index_rev_iIS312)
# Create Rev Vector:
rev_iIS312 = zeros(Bool,n_iIS312)
for i in reversible_reactions_id_iIS312
    rev_iIS312[i] = true
end
# Run QFCA on S and rev, and save the output to fctable:
fctable_QFCA_iIS312 = @time sparseQFCA.QFCA(S_iIS312, rev_iIS312)[end]
# Test that the results of QFCA are correct for the iIS312 model:
@test QFCATest_iIS312(fctable_QFCA_iIS312)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

### Consistency_Checking:

## Compare the Index of blocked reactions of TheNaiveApproach and SwiftCC

## e_coli_core

## Print a message indicating that TheNaiveApproach is being run on e_coli_core

printstyled("CC_TheNaiveApproach :\n"; color=:yellow)
printstyled("e_coli_core :\n"; color=:yellow)
# Find blocked reactions in myModel_e_coli_core using TheNaiveApproach, and save the output to blockedList_TheNaive_e_coli_core:
blockedList_TheNaive_e_coli_core = @time sparseQFCA.find_blocked_reactions(myModel_e_coli_core)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## Print a message indicating that SwiftCC is being run on e_coli_core

printstyled("CC_SwiftCC :\n"; color=:yellow)
printstyled("e_coli_core :\n"; color=:yellow)

## Get the necessary data from myModel_e_coli_core

S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, n_genes_e_coli_core, lb_e_coli_core, ub_e_coli_core, c_vector = sparseQFCA.dataOfModel(myModel_e_coli_core)
# Check for duplicate reactions in Reactions_e_coli_core:
check_duplicate = sparseQFCA.check_duplicate_reactions(Reactions_e_coli_core)
# Homogenize the lower and upper bounds of the reactions in myModel_e_coli_core:
lb_e_coli_core, ub_e_coli_core = sparseQFCA.homogenization(lb_e_coli_core, ub_e_coli_core)
# Create a ModelObject from the data in myModel_e_coli_core:
ModelObject_e_coli_core = sparseQFCA.Model_CC(S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, lb_e_coli_core, ub_e_coli_core)
# Find blocked reactions in the e_coli_core model using the swiftCC method and time the operation:
blockedList_swiftCC_e_coli_core, dualVar_e_coli_core = @time sparseQFCA.swiftCC(ModelObject_e_coli_core)
# Test that the results of the naive approach and swiftCC approach are the same:
@test blockedTest_e_coli_core(blockedList_TheNaive_e_coli_core, blockedList_swiftCC_e_coli_core)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## iIS312

## Print a message indicating that TheNaiveApproach is being run on iIS312

printstyled("CC_TheNaiveApproach :\n"; color=:yellow)
printstyled("iIS312 :\n"; color=:yellow)
# Find blocked reactions in the iIS312 model and time the operation:
blockedList_TheNaive_iIS312 = @time sparseQFCA.find_blocked_reactions(myModel_iIS312)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## Print a message indicating that SwiftCC is being run on iIS312

printstyled("CC_SwiftCC :\n"; color=:yellow)
printstyled("iIS312 :\n"; color=:yellow)

# Get data from the iIS312 model:
S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, n_genes_iIS312, lb_iIS312, ub_iIS312, c_vector = sparseQFCA.dataOfModel(myModel_iIS312)
# Check for duplicate reactions in the iIS312 model:
check_duplicate = sparseQFCA.check_duplicate_reactions(Reactions_iIS312)
# Homogenize the lower and upper bounds for reactions in the iIS312 model:
lb_iIS312, ub_iIS312 = sparseQFCA.homogenization(lb_iIS312, ub_iIS312)
# Create a model object from the iIS312 model data:
ModelObject_iIS312 = sparseQFCA.Model_CC(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, lb_iIS312, ub_iIS312)
# Find blocked reactions in the iIS312 model using the swiftCC method and time the operation:
blockedList_swiftCC_iIS312, dualVar_e_coli_core_iIS312  = @time sparseQFCA.swiftCC(ModelObject_iIS312)
# Test that the results of the naive approach and swiftCC approach are the same:
@test blockedTest_iIS312(blockedList_TheNaive_iIS312, blockedList_swiftCC_iIS312)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

### distributedQFCA:

## Print a message indicating that distributedQFCA is being run on e_coli_core

printstyled("distributedQFCA :\n"; color=:yellow)
printstyled("e_coli_core :\n"; color=:yellow)

# Extracte relevant data from input model:
S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, n_genes_e_coli_core, lb_e_coli_core, ub_e_coli_core, c_vector = sparseQFCA.dataOfModel(myModel_e_coli_core)
# Ensure that the bounds of all reactions are homogenous
lb_e_coli_core, ub_e_coli_core = sparseQFCA.homogenization(lb_e_coli_core, ub_e_coli_core)
# Separate reactions into reversible and irreversible sets:
# Create an array of reaction IDs:
Reaction_Ids_e_coli_core = collect(1:n_e_coli_core)
irreversible_reactions_id_e_coli_core, reversible_reactions_id_e_coli_core = sparseQFCA.reversibility(lb_e_coli_core, Reaction_Ids_e_coli_core)
# Create a new instance of the input model with homogenous bounds:
ModelObject_CC_e_coli_core = sparseQFCA.Model_CC(S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, lb_e_coli_core, ub_e_coli_core)
blocked_index_e_coli_core, dualVar_e_coli_core = sparseQFCA.swiftCC(ModelObject_CC_e_coli_core)
blocked_index_rev_e_coli_core = blocked_index_e_coli_core ∩ reversible_reactions_id_e_coli_core
# Convert to Vector{Int64}:
blocked_index_rev_e_coli_core = convert(Vector{Int64}, blocked_index_rev_e_coli_core)
# Correct Reversibility:
ModelObject_Crrection_e_coli_core = sparseQFCA.Model_Correction(S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, lb_e_coli_core, ub_e_coli_core, irreversible_reactions_id_e_coli_core, reversible_reactions_id_e_coli_core)
S_e_coli_core, lb_e_coli_core, ub_e_coli_core, irreversible_reactions_id_e_coli_core, reversible_reactions_id_e_coli_core = sparseQFCA.distributedReversibility_Correction(ModelObject_Crrection_e_coli_core, blocked_index_rev_e_coli_core)
# Run distributedQFCA method on the model and time the operation:
row_e_coli_core, col_e_coli_core = size(S_e_coli_core)
ModelObject_QFCA_e_coli_core = sparseQFCA.Model_QFCA(S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, row_e_coli_core, col_e_coli_core, lb_e_coli_core, ub_e_coli_core, irreversible_reactions_id_e_coli_core, reversible_reactions_id_e_coli_core)
# Run distributedQFCA method on the e_coli_core model and time the operation:
# Convert to Vector{Int64}:
blocked_index_e_coli_core = convert(Vector{Int64}, blocked_index_e_coli_core)
fctable_distributedQFCA_e_coli_core, Fc_Coefficients_e_coli_core, Dc_Coefficients_e_coli_core = @time sparseQFCA.distributedQFCA(ModelObject_QFCA_e_coli_core, blocked_index_e_coli_core)
# convert the shared matrix to a regular matrix:
fctable_distributedQFCA_e_coli_core = convert(Matrix{Int}, fctable_distributedQFCA_e_coli_core)
# Test that the results of distributedQFCA are correct for the e_coli_core model:
@test distributedQFCATest_e_coli_core(fctable_distributedQFCA_e_coli_core)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## Print a message indicating that distributedQFCA is being run on iIS312

printstyled("distributedQFCA :\n"; color=:yellow)
printstyled("iIS312 :\n"; color=:yellow)

# Extracte relevant data from input model:
S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, n_genes_iIS312, lb_iIS312, ub_iIS312, c_vector = sparseQFCA.dataOfModel(myModel_iIS312)
# Ensure that the bounds of all reactions are homogenous
lb_iIS312, ub_iIS312 = sparseQFCA.homogenization(lb_iIS312, ub_iIS312)
# Separate reactions into reversible and irreversible sets:
# Create an array of reaction IDs:
Reaction_Ids_iIS312 = collect(1:n_iIS312)
irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312 = sparseQFCA.reversibility(lb_iIS312, Reaction_Ids_iIS312)
# Create a new instance of the input model with homogenous bounds:
ModelObject_CC_iIS312 = sparseQFCA.Model_CC(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, lb_iIS312, ub_iIS312)
blocked_index_iIS312, dualVar_iIS312 = sparseQFCA.swiftCC(ModelObject_CC_iIS312)
blocked_index_rev_iIS312 = blocked_index_iIS312 ∩ reversible_reactions_id_iIS312
# Convert to Vector{Int64}:
blocked_index_rev_iIS312 = convert(Vector{Int64}, blocked_index_rev_iIS312)
# Correct Reversibility:
ModelObject_Crrection_iIS312 = sparseQFCA.Model_Correction(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, m_iIS312, n_iIS312, lb_iIS312, ub_iIS312, irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312)
S_iIS312, lb_iIS312, ub_iIS312, irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312 = sparseQFCA.distributedReversibility_Correction(ModelObject_Crrection_iIS312, blocked_index_rev_iIS312)
# Run distributedQFCA method on the model and time the operation:
row_iIS312, col_iIS312 = size(S_iIS312)
ModelObject_QFCA_iIS312 = sparseQFCA.Model_QFCA(S_iIS312, Metabolites_iIS312, Reactions_iIS312, Genes_iIS312, row_iIS312, col_iIS312, lb_iIS312, ub_iIS312, irreversible_reactions_id_iIS312, reversible_reactions_id_iIS312)
# Run distributedQFCA method on the iIS312 model and time the operation:
# Convert to Vector{Int64}:
blocked_index_iIS312 = convert(Vector{Int64}, blocked_index_iIS312)
fctable_distributedQFCA_iIS312, Fc_Coefficients_iIS312, Dc_Coefficients_iIS312 = @time sparseQFCA.distributedQFCA(ModelObject_QFCA_iIS312, blocked_index_iIS312)
# convert the shared matrix to a regular matrix:
fctable_distributedQFCA_iIS312 = convert(Matrix{Int}, fctable_distributedQFCA_iIS312)
# Test that the results of distributedQFCA are correct for the iIS312 model:
@test distributedQFCATest_iIS312(fctable_distributedQFCA_iIS312)
# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:magenta)

### ToyModels

printstyled("ToyModels:\n"; color=:magenta)

## ToyModel1

ToyModel = Model()
modelName = "ToyModel1"

printstyled("$modelName :\n"; color=:yellow)

# Genes:
for i = 1:9
    gene = "G" * "$i"
    ToyModel.genes[gene] = Gene()
end

## Metabolites

# IntraCellular:

#m1c
ToyModel.metabolites["m1"] = Metabolite(name = "M1_c", compartment = "inside")
#m2c
ToyModel.metabolites["m2"] = Metabolite(name = "M2_c", compartment = "inside")
#m3c
ToyModel.metabolites["m3"] = Metabolite(name = "M3_c", compartment = "inside")
#m4c
ToyModel.metabolites["m4"] = Metabolite(name = "M4_c", compartment = "inside")

# ExtraCellular:

#m1e
ToyModel.metabolites["m5"] = Metabolite(name = "M1_e", compartment = "outside")
#m3e
ToyModel.metabolites["m6"] = Metabolite(name = "M3_e", compartment = "outside")

## Reactions

M = sparseQFCA.getM(0)

# Forward:

ToyModel.reactions["M1t"] = Reaction(
    name = "transport m1",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m5" => -1.0, "m1" => 1.0),
    gene_association_dnf = [["G1","G2"],["G3"]],
    objective_coefficient = 0.0,
)

ToyModel.reactions["rxn2"] = Reaction(
    name = "rxn2",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m1" => -2.0, "m2" => 1.0, "m3" => 1.0),
    gene_association_dnf = [["G2"], ["G3"]],
    objective_coefficient = 0.0,
)

ToyModel.reactions["rxn3"] = Reaction(
    name = "rxn3",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0, "m3" => 1.0),
    gene_association_dnf = [["G3","G4"],["G5","G6"]],
    objective_coefficient = 0.0,
)

ToyModel.reactions["M2t"] = Reaction(
    name = "transport m2",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0, "m5" => 1.0),
    gene_association_dnf = [["G4"], ["G1","G7"], ["G3","G5"]],
    objective_coefficient = 0.0,
)

# Foward and Backward:

ToyModel.reactions["rxn1"] = Reaction(
    name = "rxn1",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m1" => -1.0, "m4" => 1.0),
    gene_association_dnf = [["G9"]],
    objective_coefficient = 0.0,
)

ToyModel.reactions["M3t"] = Reaction(
    name = "transport m3",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m3" => -1.0, "m6" => 1.0),
    gene_association_dnf = [["G6"]],
    objective_coefficient = 0.0,
)

# Exchange:

ToyModel.reactions["EX_1"] = Reaction(
    name = "exchange m5",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m5" => -1.0),
    gene_association_dnf = [["G7"]],
    objective_coefficient = 0.0,
)

ToyModel.reactions["EX_2"] = Reaction(
    name = "exchange m6",
    lower_bound = -20,
    upper_bound = M,
    stoichiometry = Dict("m6" => -1.0),
    gene_association_dnf = [["G8"]],
    objective_coefficient = 1.0,
)

ToyModel_json = convert(JSONFBCModel, ToyModel)
save_model(ToyModel_json, "../test/Models/$modelName.json")  # Use the string in the file path
# Read the JSON file
data = JSON.parsefile("Models/$modelName.json")

# Process reactions to replace '&&' with 'and' and '||' with 'or' in gene_reaction_rule
if haskey(data, "reactions")
    for reaction in data["reactions"]
        if haskey(reaction, "gene_reaction_rule") && !isempty(reaction["gene_reaction_rule"])
            reaction["gene_reaction_rule"] = replace(reaction["gene_reaction_rule"], "&&" => "and", "||" => "or")
        end
    end
end

# Write the corrected JSON file
open("Models/$modelName.json", "w") do file
    JSON.print(file, data, 1)  # Use 'indent=1' for indentation
end

S_ToyModel, Metabolites_ToyModel, Reactions_ToyModel, Genes_ToyModel, m_ToyModel, n_ToyModel, n_genes_ToyModel, lb_ToyModel, ub_ToyModel, c_vector_ToyModel = sparseQFCA.dataOfModel(ToyModel)

## FBA

V_initial, Original_ObjectiveValue = sparseQFCA.FBA(ToyModel, modelName)

## Corrected FBA

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(ToyModel, modelName)

## QuantomeRedNet

ModelName = "ToyModel1"
myModel_ToyModel = load_model(JSONFBCModel, "Models/$modelName.json", A.CanonicalModel.Model)

printstyled("QuantomeRedNet - $modelName :\n"; color=:yellow)

compressedModelName, A_matrix, compression_map = sparseQFCA.quantomeReducer(myModel_ToyModel, ModelName, "HiGHS", false, false)

ToyModel_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/$compressedModelName.json", A.CanonicalModel.Model)

S_ToyModelcompressed, Metabolites_ToyModelcompressed, Reactions_ToyModelcompressed, Genes_ToyModelcompressed, m_ToyModelcompressed, n_ToyModelcompressed, n_genes_ToyModelcompressed, lb_ToyModelcompressed, ub_ToyModelcompressed, c_vector_ToyModelcompressed = sparseQFCA.dataOfModel(ToyModel_compressed, 0)

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(ToyModel, ToyModel_compressed, A_matrix, modelName)

@test FBATest(Original_ObjectiveValue, Corrected_ObjectiveValue, Compressed_ObjectiveValue)

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

## ToyModel2

printstyled("ToyModel2:\n"; color=:yellow)

ToyModel2 = Model()
modelName = "ToyModel2"

# Genes:
ToyModel2.genes["G1"] = Gene()

## Metabolites

# IntraCellular:

#m1c
ToyModel2.metabolites["m1"] = Metabolite(name = "M1_c", compartment = "inside")

## Reactions

M = sparseQFCA.getM(0)

ToyModel2.reactions["rxn1"] = Reaction(
    name = "rxn1",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m1" => 2.0),
    gene_association_dnf = [["G1"]],
    objective_coefficient = 0.0,
)

ToyModel2.reactions["rxn2"] = Reaction(
    name = "rxn2",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m1" => -1.0),
    gene_association_dnf = [],
    objective_coefficient = 1.0,
)

ToyModel2_json = convert(JSONFBCModel, ToyModel2)
save_model(ToyModel2_json, "../test/Models/$modelName.json")  # Use the string in the file path
# Read the JSON file
data = JSON.parsefile("Models/$modelName.json")

# Process reactions to replace '&&' with 'and' and '||' with 'or' in gene_reaction_rule
if haskey(data, "reactions")
    for reaction in data["reactions"]
        if haskey(reaction, "gene_reaction_rule") && !isempty(reaction["gene_reaction_rule"])
            reaction["gene_reaction_rule"] = replace(reaction["gene_reaction_rule"], "&&" => "and", "||" => "or")
        end
    end
end

# Write the corrected JSON file
open("Models/$modelName.json", "w") do file
    JSON.print(file, data, 1)  # Use 'indent=1' for indentation
end

S_ToyModel2, Metabolites_ToyModel2, Reactions_ToyModel2, Genes_ToyModel2, m_ToyModel2, n_ToyModel2, n_genes_ToyModel2, lb_ToyModel2, ub_ToyModel2, c_vector_ToyModel2 = sparseQFCA.dataOfModel(ToyModel2)

## FBA

V_initial, Original_ObjectiveValue = sparseQFCA.FBA(ToyModel2, modelName)

## Corrected FBA

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(ToyModel2, modelName)

## QuantomeRedNet

myModel_ToyModel2 = load_model(JSONFBCModel, "Models/$modelName.json", A.CanonicalModel.Model)

printstyled("QuantomeRedNet - $modelName :\n"; color=:yellow)

ModelName = "ToyModel2"

compressedModelName, A_matrix, compression_map = sparseQFCA.quantomeReducer(myModel_ToyModel2, ModelName, "HiGHS", false, false)

ToyModel2_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/$compressedModelName.json", A.CanonicalModel.Model)

S_ToyModel2compressed, Metabolites_ToyModel2compressed, Reactions_ToyModel2compressed, Genes_ToyModel2compressed, m_ToyModel2compressed, n_ToyModel2compressed, n_genes_ToyModel2compressed, lb_ToyModel2compressed, ub_ToyModel2compressed, c_vector_ToyModel2compressed = sparseQFCA.dataOfModel(ToyModel2_compressed, 0)

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(ToyModel2, ToyModel2_compressed, A_matrix, modelName)

@test FBATest(Original_ObjectiveValue, Corrected_ObjectiveValue, Compressed_ObjectiveValue)

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

# ToyModel3

ToyModel3 = Model()
modelName = "ToyModel3"

printstyled("$modelName:\n"; color=:yellow)

# Genes:
for i = 1:5
    gene = "G" * "$i"
    ToyModel3.genes[gene] = Gene()
end

## Metabolites

# IntraCellular:

#m1c
ToyModel3.metabolites["m1"] = Metabolite(name = "M1_c", compartment = "inside")

#m2c
ToyModel3.metabolites["m2"] = Metabolite(name = "M2_c", compartment = "inside")

#m3c
ToyModel3.metabolites["m3"] = Metabolite(name = "M3_c", compartment = "inside")

## Reactions

M = sparseQFCA.getM(0)

# Forward:

ToyModel3.reactions["R1"] = Reaction(
    name = "rxn1",
    lower_bound = -10,
    upper_bound = M,
    stoichiometry = Dict("m1" => -1.0),
    gene_association_dnf = [["G1"]],
    objective_coefficient = 0.0,
)

ToyModel3.reactions["R2"] = Reaction(
    name = "rxn2",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m1" => -2.0, "m2" => 1.0),
    gene_association_dnf = [["G2"]],
    objective_coefficient = 0.0,
)

ToyModel3.reactions["R3"] = Reaction(
    name = "rxn3",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0),
    gene_association_dnf = [["G3"]],
    objective_coefficient = 0.0,
)

ToyModel3.reactions["R4"] = Reaction(
    name = "rxn4",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0, "m3" => 1.0),
    gene_association_dnf = [["G4"]],
    objective_coefficient = 0.0,
)

ToyModel3.reactions["R5"] = Reaction(
    name = "rxn5",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m3" => -1.0),
    gene_association_dnf = [["G5"]],
    objective_coefficient = 1.0,
)

ToyModel3_json = convert(JSONFBCModel, ToyModel3)
save_model(ToyModel3_json, "../test/Models/$modelName.json")  # Use the string in the file path
# Read the JSON file
data = JSON.parsefile("Models/$modelName.json")

# Process reactions to replace '&&' with 'and' and '||' with 'or' in gene_reaction_rule
if haskey(data, "reactions")
    for reaction in data["reactions"]
        if haskey(reaction, "gene_reaction_rule") && !isempty(reaction["gene_reaction_rule"])
            reaction["gene_reaction_rule"] = replace(reaction["gene_reaction_rule"], "&&" => "and", "||" => "or")
        end
    end
end

# Write the corrected JSON file
open("Models/$modelName.json", "w") do file
    JSON.print(file, data, 1)  # Use 'indent=1' for indentation
end

S_ToyModel3, Metabolites_ToyModel3, Reactions_ToyModel3, Genes_ToyModel3, m_ToyModel3, n_ToyModel3, n_genes_ToyModel3, lb_ToyModel3, ub_ToyModel3, c_vector_ToyModel3 = sparseQFCA.dataOfModel(ToyModel3)

## FBA

V_initial, Original_ObjectiveValue = sparseQFCA.FBA(ToyModel3, modelName)

## Corrected FBA

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(ToyModel3, modelName)

## QuantomeRedNet

myModel_ToyModel3 = load_model(JSONFBCModel, "Models/$modelName.json", A.CanonicalModel.Model)

printstyled("QuantomeRedNet - $modelName :\n"; color=:yellow)

ModelName = "ToyModel3"

compressedModelName, A_matrix, compression_map = sparseQFCA.quantomeReducer(myModel_ToyModel3, ModelName, "HiGHS", false, false)

ToyModel3_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/$compressedModelName.json", A.CanonicalModel.Model)

S_ToyModel3compressed, Metabolites_ToyModel3compressed, Reactions_ToyModel3compressed, Genes_ToyModel3compressed, m_ToyModel3compressed, n_ToyModel3compressed, n_genes_ToyModel3compressed, lb_ToyModel3compressed, ub_ToyModel3compressed, c_vector_ToyModel3compressed = sparseQFCA.dataOfModel(ToyModel3_compressed, 0)

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(ToyModel3, ToyModel3_compressed, A_matrix, modelName)

@test FBATest(Original_ObjectiveValue, Corrected_ObjectiveValue, Compressed_ObjectiveValue)

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

# ToyModel4

ToyModel4 = Model()
modelName = "ToyModel4"

printstyled("$modelName:\n"; color=:yellow)

# Genes:
for i = 1:7
    gene = "G" * "$i"
    ToyModel4.genes[gene] = Gene()
end

## Metabolites

# IntraCellular:

#m1c
ToyModel4.metabolites["m1"] = Metabolite(name = "M1_c", compartment = "inside")

# ExtraCellular:

#m1e
ToyModel4.metabolites["m2"] = Metabolite(name = "M1_e", compartment = "outside")
#m2e
ToyModel4.metabolites["m3"] = Metabolite(name = "M2_e", compartment = "outside")
#m3e
ToyModel4.metabolites["m4"] = Metabolite(name = "M3_e", compartment = "outside")

## Reactions

M = sparseQFCA.getM(0)

ToyModel4.reactions["M1t"] = Reaction(
    name = "transport m1",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0, "m1" => 1.0),
    gene_association_dnf = [["G1"]],
    objective_coefficient = 0.0,
)

ToyModel4.reactions["M2t"] = Reaction(
    name = "transport m2",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m1" => -1.0, "m3" => 1.0),
    gene_association_dnf = [["G2"]],
    objective_coefficient = 0.0,
)

ToyModel4.reactions["M3t"] = Reaction(
    name = "transport m3",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m1" => -1.0, "m4" => 1.0),
    gene_association_dnf = [["G3"]],
    objective_coefficient = 0.0,
)

# Exchange:

ToyModel4.reactions["EX_1"] = Reaction(
    name = "exchange M1e",
    lower_bound = -5,
    upper_bound = M,
    stoichiometry = Dict("m2" => -1.0),
    gene_association_dnf = [["G4"]],
    objective_coefficient = 0.0,
)

ToyModel4.reactions["EX_2"] = Reaction(
    name = "exchange M2e",
    lower_bound = -7,
    upper_bound = M,
    stoichiometry = Dict("m3" => -1.0),
    gene_association_dnf = [["G5"]],
    objective_coefficient = 0.0,
)

ToyModel4.reactions["EX_3"] = Reaction(
    name = "exchange M3e",
    lower_bound = -M,
    upper_bound = M,
    stoichiometry = Dict("m4" => -1.0),
    gene_association_dnf = [["G6"]],
    objective_coefficient = 1.0,
)

ToyModel4.reactions["rxn1"] = Reaction(
    name = "rxn1",
    lower_bound = 0.0,
    upper_bound = M,
    stoichiometry = Dict("m3" => -1.0, "m2" => 1.0),
    gene_association_dnf = [["G7"]],
    objective_coefficient = 0.0,
)

ToyModel4_json = convert(JSONFBCModel, ToyModel4)
save_model(ToyModel4_json, "../test/Models/$modelName.json")  # Use the string in the file path
# Read the JSON file
data = JSON.parsefile("Models/$modelName.json")

# Process reactions to replace '&&' with 'and' and '||' with 'or' in gene_reaction_rule
if haskey(data, "reactions")
    for reaction in data["reactions"]
        if haskey(reaction, "gene_reaction_rule") && !isempty(reaction["gene_reaction_rule"])
            reaction["gene_reaction_rule"] = replace(reaction["gene_reaction_rule"], "&&" => "and", "||" => "or")
        end
    end
end

# Write the corrected JSON file
open("Models/$modelName.json", "w") do file
    JSON.print(file, data, 1)  # Use 'indent=1' for indentation
end

S_ToyModel4, Metabolites_ToyModel4, Reactions_ToyModel4, Genes_ToyModel4, m_ToyModel4, n_ToyModel4, n_genes_ToyModel4, lb_ToyModel4, ub_ToyModel4, c_vector_ToyModel4 = sparseQFCA.dataOfModel(ToyModel4)

## FBA

V_initial, Original_ObjectiveValue = sparseQFCA.FBA(ToyModel4, modelName)

## Corrected FBA

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(ToyModel4, modelName)

## QuantomeRedNet

myModel_ToyModel4 = load_model(JSONFBCModel, "Models/$modelName.json", A.CanonicalModel.Model)

printstyled("QuantomeRedNet - $modelName :\n"; color=:yellow)

ModelName = "ToyModel4"

compressedModelName, A_matrix, compression_map = sparseQFCA.quantomeReducer(myModel_ToyModel4, ModelName, "HiGHS", false, false)

ToyModel4_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/$compressedModelName.json", A.CanonicalModel.Model)

S_ToyModel4compressed, Metabolites_ToyModel4compressed, Reactions_ToyModel4compressed, Genes_ToyModel4compressed, m_ToyModel4compressed, n_ToyModel4compressed, n_genes_ToyModel4compressed, lb_ToyModel4compressed, ub_ToyModel4compressed, c_vector_ToyModel4compressed = sparseQFCA.dataOfModel(ToyModel4_compressed, 0)

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(ToyModel4, ToyModel4_compressed, A_matrix, modelName)

@test FBATest(Original_ObjectiveValue, Corrected_ObjectiveValue, Compressed_ObjectiveValue)

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:magenta)

### Metabolic Network Compression

printstyled("Metabolic Network Compression:\n"; color=:magenta)

modelName = "e_coli_core"

# Extracte relevant data from input model:

S_e_coli_core, Metabolites_e_coli_core, Reactions_e_coli_core, Genes_e_coli_core, m_e_coli_core, n_e_coli_core, n_genes_e_coli_core, lb_e_coli_core, ub_e_coli_core, c_vector_e_coli_core = sparseQFCA.dataOfModel(myModel_e_coli_core)

## FBA

V_initial, Original_ObjectiveValue = sparseQFCA.FBA(myModel_e_coli_core, modelName)

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## Corrected FBA

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(myModel_e_coli_core, modelName)

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## QuantomeRedNet

printstyled("QuantomeRedNet - $modelName :\n"; color=:yellow)

compressedModelName, A_matrix, compression_map = sparseQFCA.quantomeReducer(myModel_e_coli_core, ModelName, "HiGHS", true, false)

e_coli_core_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/$compressedModelName.json", A.CanonicalModel.Model)

S_e_coli_core_compressed, Metabolites_e_coli_core_compressed, Reactions_e_coli_core_compressed, Genes_e_coli_core_compressed, m_e_coli_core_compressed, n_e_coli_core_compressed, n_genes_e_coli_core_compressed, lb_e_coli_core_compressed, ub_e_coli_core_compressed, c_vector_e_coli_core_compressed = sparseQFCA.dataOfModel(e_coli_core_compressed, 0)

e_coli_core = load_model(JSONFBCModel, "Models/e_coli_core.json", A.CanonicalModel.Model)

result_compress, time_taken_compress, bytes_alloc_compress, gctime_compress = @timed begin

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(e_coli_core, e_coli_core_compressed, A_matrix, modelName)

end

@test FBATest(Original_ObjectiveValue, Corrected_ObjectiveValue, Compressed_ObjectiveValue)

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

### distributedFBA

using COBRA

#workersPool, nWorkers = createPool(8, false)

model = loadModel("Models/e_coli_core.mat", "e_coli_core")

result_distributedFBA, time_taken_distributedFBA, bytes_alloc_distributedFBA, gctime_distributedFBA = @timed begin
## set the reaction list

rxnsList = 1:n_e_coli_core

## select the reaction optimization mode
##  0: only minimization
##  1: only maximization
##  2: maximization and minimization

rxnsOptMode = 1

## specify the solver name

solverName = :GLPK

# set solver parameters
    solParams = [
        # decides whether or not results are displayed on screen in an application of the C API.
        (:CPX_PARAM_SCRIND,         0);

        # sets the parallel optimization mode. Possible modes are automatic, deterministic, and opportunistic.
        (:CPX_PARAM_PARALLELMODE,   1);

        # sets the default maximal number of parallel threads that will be invoked by any CPLEX parallel optimizer.
        (:CPX_PARAM_THREADS,        1);

        # partitions the number of threads for CPLEX to use for auxiliary tasks while it solves the root node of a problem.
        (:CPX_PARAM_AUXROOTTHREADS, 2);

        # decides how to scale the problem matrix.
        (:CPX_PARAM_SCAIND,         -1);

        # controls which algorithm CPLEX uses to solve continuous models (LPs).
        (:CPX_PARAM_LPMETHOD,       0)
    ] #end of solParams

## change the COBRA solver

solver = changeCobraSolver(solverName, solParams)

# preFBA

optSol, fbaSol = preFBA!(model, solver)

# spliteRange

splitRange(model, rxnsList, 4, 1)

# loopFBA

m, x, c = buildCobraLP(model, solver)

retObj, retFlux, retStat = loopFBA(m, x, c, rxnsList, n_e_coli_core)

minFlux, maxFlux = distributedFBA(model, solver)

end

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

## COBREXA

import JSONFBCModels
import HiGHS

model = load_model("Models/e_coli_core.json")

result, time_taken_COBREXA, bytes_alloc_COBREXA, gctime_COBREXA = @timed begin

solution = flux_balance_analysis(model, optimizer = HiGHS.Optimizer)
println("COBREXA FBA:")
println("Biomass Flux = $(solution.objective)")

end

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

println("compressedFBA:")
println("Time: ", time_taken_compress, " seconds")
println("Memory Allocations: ", bytes_alloc_compress / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_compress, " seconds")

println("distrubtedFBA:")
println("Time: ", time_taken_distributedFBA, " seconds")
println("Memory Allocations: ", bytes_alloc_distributedFBA / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_distributedFBA, " seconds")

println("COBREXA FBA:")
println("Time: ", time_taken_COBREXA, " seconds")
println("Memory Allocations: ", bytes_alloc_COBREXA / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_COBREXA, " seconds")

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:magenta)

### Compressed Flux Balance Analysis

printstyled("Compressed Flux Balance Analysis by using CompressedModel:\n"; color=:magenta)

## e_coli_core

modelName = "e_coli_core"
myModel_e_coli_core2 = load_model(JSONFBCModel, "Models/e_coli_core.json", A.CanonicalModel.Model)
V_initial, Original_ObjectiveValue = sparseQFCA.FBA(myModel_e_coli_core2, modelName)

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

V_correction, Corrected_ObjectiveValue = sparseQFCA.correctedFBA(myModel_e_coli_core2, modelName)

printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:yellow)

## Load Compressed Model

myModel_e_coli_core2_compressed = load_model(JSONFBCModel, "../src/QuantomeRedNet/CompressionResults/e_coli_core_compressed.json", A.CanonicalModel.Model)

## Load A Matrix

jld_file = jldopen("../src/QuantomeRedNet/CompressionResults/A_e_coli_core_compressed.jld2", "r")
    A_matrix = jld_file["A"]
close(jld_file)

result_compress, time_taken_compress, bytes_alloc_compress, gctime_compress = @timed begin

V, V_compressed, Compressed_ObjectiveValue = sparseQFCA.compressedFBA(myModel_e_coli_core2, myModel_e_coli_core2_compressed, A_matrix, modelName)

end

println("compressedFBA:")
println("Time: ", time_taken_compress, " seconds")
println("Memory Allocations: ", bytes_alloc_compress / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_compress, " seconds")

println("distrubtedFBA:")
println("Time: ", time_taken_distributedFBA, " seconds")
println("Memory Allocations: ", bytes_alloc_distributedFBA / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_distributedFBA, " seconds")

println("COBREXA FBA:")
println("Time: ", time_taken_COBREXA, " seconds")
println("Memory Allocations: ", bytes_alloc_COBREXA / (1024^2), " MB")
println("Garbage Collection Time: ", gctime_COBREXA, " seconds")

# Print a separator:
printstyled("#-------------------------------------------------------------------------------------------#\n"; color=:red)

### End
