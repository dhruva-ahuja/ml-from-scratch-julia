function sgd_linear_regression(X, Y)
    n = length(X)
    m = rand()
    b = rand()

    learning_rate = 0.01
    epochs = 10000

    for _ in 1:epochs
        random_index = rand(1:n)
        x_i = X[random_index]
        y_i = Y[random_index]

        pred = x_i * m + b

        grad_m = 2 * (pred - y_i) * x_i
        grad_b = 2 * (pred - y_i)

        m -= learning_rate * grad_m
        b -= learning_rate * grad_b
    end

    m, b
end

@main function main(_)
    X = [0.7, 1.5, 2.1, 2.9, 3.3, 4.5]
    Y = [150, 300, 320, 360, 400, 480]
    (m, b) = sgd_linear_regression(X, Y)
    println(m, ' ', b)
end
