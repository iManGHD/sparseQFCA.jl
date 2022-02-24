module pre_processing

export dataOfModel, setM, reversibility, check_duplicate_reaction,
       homogenization, reversibility_checking, reversibility_correction

using COBREXA, JuMP, GLPK

#=
    dataOfModel
        input: model
        output: S, Metabolites, Reactions, Genes,
                #Row of S (m) = #Metabolites,
                #Column of S (n) = #Reactions,
                LowerBound Of Reactions(lb),
                UpperBounds of Reactions(ub)
=#

function dataOfModel(myModel)
    S = stoichiometry(myModel)
    Metabolites = metabolites(myModel)
    Reactions = reactions(myModel)
    Genes = genes(myModel)
    m = length(metabolites(myModel))
    n = length(reactions(myModel))
    lb = lower_bounds(myModel)
    ub = upper_bounds(myModel)
    return S, Metabolites, Reactions, Genes, m, n, lb, ub
end

#=
    setM
        input: A large number
        output: M has been set
=#

function setM(x)
    global M = x
    return
end

#=
    Reversibility
        input: lb
        output: irreversible_reactions_id, reversible_reactions_id
=#

function reversibility(lb)
    n = length(lb)
    irreversible_reactions_id = []
    reversible_reactions_id = []
    for i in 1:n
        if lb[i] >= 0
            append!(irreversible_reactions_id, i)
        else
            append!(reversible_reactions_id, i)
        end
    end
    return irreversible_reactions_id, reversible_reactions_id
end

#=
    check_duplicate_reaction
        input: Reactions
        output:
        True: There are a number of repetitive reactions
        Flase: There is no repetitive reaction
=#

function check_duplicate_reaction(Reactions)
    n = length(Reactions)
    unique_reactions = unique!(Reactions)
    n_unique = length(unique_reactions)
    if n == n_unique
        return false
    else
        return true
    end
end

#=
    homogenization
        input: lb, ub
        output: modified lb, ub
=#

function homogenization(lb, ub)
    n = length(lb)
    for i in 1:n
        if lb[i] > 0
            lb[i] = 0
        end
        if ub[i] > 0
            ub[i] = M
        end
        if lb[i] < 0
            lb[i] = -M
        end
        if ub[i] < 0
            ub[i] = 0
        end
    end
    return lb, ub
end

#=
    reversibility_Checking
        input: reversible_reactions_id, lb
        output:
                 rev_blocked_fwd : Reversible reactions that are blocked in Forward Direction
                 rev_blocked_back : Reversible reactions that are blocked in Backward Direction
=#

function reversibility_checking(reversible_reactions_id, lb)
    
    n = length(lb)
    model = Model(GLPK.Optimizer)
    @variable(model, lb[i] <= V[i = 1:n] <= ub[i])
    @constraint(model, S * V .== 0)
    rev_blocked_fwd = []
    rev_blocked_back = []

    for j in reversible_reactions_id

    # The forward direction:

        @objective(model, Max, V[j])
        @constraint(model, V[j] <= 1)
        optimize!(model)
        opt_fwd = objective_value(model)
    
        if opt_fwd ≈ 0
             append!(rev_blocked_fwd, j)
        end

    # The backward direction:

        @objective(model, Min, V[j])
        @constraint(model, V[j] >= -1)
        optimize!(model)
        opt_back = objective_value(model)
    
        if opt_back ≈ 0
             append!(rev_blocked_back, j)
        end
     end
    return rev_blocked_fwd, rev_blocked_back
end

#=
    reversibility_correction

        input:  S, lb, ub, irreversible_reactions_id, reversible_reactions_id,
                  rev_blocked_fwd, rev_blocked_back
        output:  Modified S, lb, ub, irreversible_reactions_id, reversible_reactions_id, 
                  rev_blocked_fwd, rev_blocked_back
=#

function reversibility_correction(S, lb, ub, irreversible_reactions_id, reversible_reactions_id, rev_blocked_fwd, rev_blocked_back)
    
    corrected_reversible_reactions_id = []
    
    # Forward
    
    for i in rev_blocked_fwd
    # Modify lb, ub:
        ub[i] = lb[i] * -1
        lb[i] = 0.0
    # Add to irreversible reactions list:
        append!(irreversible_reactions_id, i)
    # Modify S :
        S[:, i] .= S[:, i] * -1
    end

    # Backward

    for i in rev_blocked_back    
    # Modify lb, ub:
        lb[i] = 0.0
    # Add to irreversible reactions list:
        append!(irreversible_reactions_id, i)
    end
  
    # Remove rev_blocked_fwd and rev_blocked_back from reversible reactions list
    
    set_reversible_reactions_id = Set(reversible_reactions_id)
    set_rev_blocked_fwd = Set(rev_blocked_fwd)
    set_rev_blocked_back = Set(rev_blocked_back)
    set_rev_blocked_onedirection = union(set_rev_blocked_fwd, set_rev_blocked_back)
    set_reversible_reactions_id = setdiff(set_reversible_reactions_id, set_rev_blocked_onedirection)
    
    for i in set_reversible_reactions_id
        append!(corrected_reversible_reactions_id, i)
    end
    return S, lb, ub, irreversible_reactions_id, corrected_reversible_reactions_id
end

end