using Base: @main
using DataFrames
import StatsBase: countmap

function prior_probability(y::Vector{String})
    value_counts = countmap(y)
    n = length(y)
    Dict(map(x -> (x[1], x[2] / n), collect(value_counts)))
end

function smooth_probability(y::Vector{String})
    value_counts = countmap(y)
    n = length(y) + length(unique(y))
    Dict(map(x -> (x[1], (x[2] + 1) / n), collect(value_counts)))
end

function calculate_likelihoods(X::DataFrame, y::Vector)
    likelihoods = Dict()
    for col in names(X)
        likelihoods[col] = Dict()
        for cls in unique(y)
            class_data = X[y.==cls, col]
            likelihoods[col][cls] = smooth_probability(class_data)
        end
    end
    likelihoods
end

function naive_bayes_classifier(X_test, priors, likelihoods)
    probabilities = []
    for x in eachrow(X_test)
        class_prob = Dict()
        for class in keys(priors)
            prob = 1
            for feature in names(x)
                likelihood = likelihoods[feature][class]
                prob *= (get(likelihood, x[Symbol(feature)], 1 / (length(likelihood) + 1)))
            end
            class_prob[class] = prob
        end
        push!(probabilities, argmax(class_prob))
    end
    probabilities
end


@main function main(_)
    df = DataFrame(
        :Temperature => ["Hot", "Cold", "Cold", "Hot", "Cold"],
        :Outlook => ["Sunny", "Rainy", "Rainy", "Sunny", "Sunny"],
        :Play => ["Yes", "Yes", "No", "Yes", "No"]
    )

    new_day = DataFrame(:Temperature => "Hot", :Outlook => "Sunny")
    println(naive_bayes_classifier(new_day, prior_probability(df.Play), calculate_likelihoods(df[:, [:Temperature, :Outlook]], df.Play)))
end
