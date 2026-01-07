import Random: randperm


function mean_absolute_error(y, y_pred)
    return sum(abs.(y - y_pred)) / (length(y))
end


function mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=16, epochs=100)
    m, n = size(X)
    theta = randn(n, 1)

    for _ in 1:epochs
        shuffled_index = randperm(m)
        X_shuffled = X[shuffled_index, :]
        y_shuffled = y[shuffled_index]

        for i in 1:batch_size:m-batch_size
            Xᵢ = X_shuffled[i : i+batch_size-1, :]
            yᵢ = y_shuffled[i : i+batch_size-1]

            gradient = (2 / batch_size) * Xᵢ' * (Xᵢ * theta - yᵢ)
            theta -= learning_rate * gradient
        end
    end

    return theta
end


X = rand(100, 3)
y = 5X[:, 1] .- 3X[:, 2] .+ 2X[:, 3] .+ randn(100, 1)
θ = mini_batch_gradient_descent(X, y)
mean_absolute_error(X*θ, y)
