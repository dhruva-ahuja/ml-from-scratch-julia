import Plots: plot, plot!


# Assuming f is the cost function
f(x) = x^2;
Δf(x) = 2x;


@main function main(_)
    α = 0.9
    learning_rate = 0.01
    momentum = 0
    epochs = 50

    θ = 4 # Plain Parameter
    Θ = 4 # Momentum Parameter

    history_plain, history_momentum = [], []

    for _ in 1:epochs
        append!(history_plain, θ)
        gradient = Δf(θ)
        θ -= learning_rate * gradient
        
        append!(history_momentum, Θ)
        gradient = Δf(Θ)
        momentum = learning_rate * gradient + α * momentum
        Θ -= momentum
    end

    plot(eachindex(history_plain), f.(history_plain))
    plot!(eachindex(history_momentum), f.(history_momentum), title = "Gradient Descent vs Momentum GD", xlabel = "Epoch", ylabel = "Cost", legend = false)
end