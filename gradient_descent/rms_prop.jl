f(x, y) = (x-3)^2 + (y+1)^2
∇f(x, y) = [2(x-3), 2(y+1)]


function RMSProp(α, ρ, ϵ, grad, s_prev)
    s = ρ * s_prev + (1 - ρ) * (grad .^ 2)
    updates = α * grad ./ .√(s .+ ϵ) 
    return updates, s
end


@main function main(_)
    coordinates = [5.0, 4.0]
    s_prev = [0.0, 0.0]
    α = 0.1 # Learning rate
    ρ = 0.9 # Decay factor
    ϵ = 1e-6

    max_epochs = 100

    for epoch in 1:max_epochs
        grad = ∇f(coordinates...)
        updates, s_prev = RMSProp(α, ρ, ϵ, grad, s_prev)
        coordinates -= updates
        if epoch % 20 == 0
            println("Epoch $epoch, current state: $coordinates")
        end 
    end
end