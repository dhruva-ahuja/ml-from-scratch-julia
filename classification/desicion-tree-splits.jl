using Base: @main


function split_groups(dataset, index, value)
    left = []
    right = []
    for d in dataset
        if d[index] <= value
            push!(left, d)
        else
            push!(right, d)
        end
    end
    return left, right
end


function gini_index(groups::Vector, class_idx::Integer, classes::Set{String})
    gini = 0.0
    total_instances = sum(length(g) for g in groups)
    for g in groups
        group_instances = length(g)
        score = 0
        for cls in classes
            p = (filter(x -> x[class_idx] == cls, g) |> length) / group_instances
            score += p * p
        end
        gini += (group_instances / total_instances) * (1 - score)
    end
    gini
end


function split_dataset(dataset, class_idx)
    classes = map(x -> x[class_idx], dataset) |> Set
    b_score = Inf
    b_col = NaN
    b_val = NaN

    for idx in eachindex(dataset[1])
        if idx == class_idx
            continue
        end
        for row in dataset
            left, right = split_groups(dataset, idx, row[idx])
            gini = gini_index([left, right], class_idx, classes)
            if gini < b_score
                b_score = gini
                b_col = idx
                b_val = row[idx]
            end
        end
    end

    return b_col, b_val, b_score
end


@main function main(_)
    dataset = [[16, "Comedy", "Yes"], [21, "Action", "No"], [25, "Comedy", "Yes"]]
    class_label_index = 3
    split_dataset(dataset, class_label_index)
end
