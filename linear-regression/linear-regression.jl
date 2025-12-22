using Base: @main
import Statistics: mean, std

function gradient_descent(X::Matrix, y::Vector, θ::Vector, α::AbstractFloat, iterations::Integer)
    m = length(y)
    for _ in 1:iterations
        θ -= (α / m) * X' * (X * θ - y) # Do not forget the 1/m term as it comes in error
    end
    return θ
end


function predict(X::Matrix, θ::Vector)
    return X * θ
end


@main function main(_)
    house_sizes = [1000; 1500; 2000]
    house_prices = [300; 450; 600]

    # Important step as it made the coefficents non NaN
    house_size_norm = (house_sizes .- mean(house_sizes)) ./ std(house_sizes)
    theta = randn(2)
    learning_rate = 0.01
    iterations = 500

    X_b = hcat(Int32.(ones(3)), house_size_norm)
    θ_bar = gradient_descent(X_b, house_prices, theta, learning_rate, iterations)
    predict(X_b, θ_bar) |> println
end
