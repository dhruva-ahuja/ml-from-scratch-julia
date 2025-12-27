using StatsBase: @main

function gini_index(groups, classes)
    n_instances = sum([length(g) for g in groups])
    gini = 0
    for group in groups
        n = length(group)
        if n == 0
            continue
        end
        score = 0.0
        for class in classes
            class_len = length(filter(x -> x[3] == class, group))
            score += (class_len / n)^2
        end
        gini += n / n_instances * (1 - score)
    end
    return gini
end

@main function main(_)
    groups = [[[23, "Comedy", "Yes"], [27, "Comedy", "No"]], [[35, "Drama", "Yes"]]]
    classes = ["Yes", "No"]
    print(gini_index(groups, classes))
end
