using Base: @main
using Distributions: MvNormal, randn
import Plots: scatter

function sigmoid(z::AbstractVector)
    return 1 ./ (1 .+ exp.(-z))
end

function logistics_regression(X, y, num_iter, learning_rate)
    m = size(X)[1]
    X_b = hcat(ones(m), X)
    β = randn(Float32, size(X_b)[2])

    for _ in 1:num_iter
        p = sigmoid(X_b * β)
        β = β .- (X_b' * (p .- y)) .* (learning_rate / m)
    end

    return β
end

function predict_prod(X, β)
    m = size(X)[1]
    return sigmoid(hcat(ones(m), X) * β)
end

function predict(X, β; threshold=0.5)
    return predict_prod(X, β) .> threshold
end


@main function main(_)
    num_obs = 1000
    x1 = rand(MvNormal([0, 0], [1 0.75; 0.75 1]), num_obs)'
    x2 = rand(MvNormal([1, 4], [1 0.75; 0.75 1]), num_obs)'

    X = vcat(x1, x2)
    y = vcat(zeros(num_obs), ones(num_obs))
    β = logistics_regression(X, y, 3000, 0.1)
    pred = predict(X, β)

    accuracy = ((y .== pred) |> sum) / size(X)[1]
    println(accuracy)
end
