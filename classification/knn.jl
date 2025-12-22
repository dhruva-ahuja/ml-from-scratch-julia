using Base: @main
import DataStructures: counter

function k_nearest_neighbors(data, query, k, distance_fn)
    distances = map(x -> (distance_fn(x[1], query), x[2]), data)
    sort!(distances, rev=true)
    c = counter(map(x -> x[2], distances[1:k]))
    sort!(collect(c), by=x -> x[2], rev=true)[1][1]
end

@main function main(_)
    cosmic_objects = [
        ((1, 5), "Dwarf Star"),
        ((3, 8), "Giant Star"),
        ((2, 6), "Dwarf Star"),
    ]

    new_object = (2, 7)
    euclidian_distance = (x, y) -> [(x1 - y1)^2 for (x1, y1) in zip(x, y)] |> sum |> sqrt
    println(k_nearest_neighbors(cosmic_objects, new_object, 3, euclidian_distance))
end
