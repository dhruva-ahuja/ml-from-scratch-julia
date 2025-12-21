import Random: seed!
import LinearAlgebra: I
using Distributions: MvNormal

seed!(42)

truth_labels = [rand() > 0.6 ? true : false for _ in 1:100]

predicted_probs = rand(MvNormal(truth_labels, 0.3 * I(length(truth_labels))))
predicted_probs = min.(max.(0, predicted_probs), 1)


function roc_curve(truth_labels, predicted_probs)
    tprs = []
    fprs = []
    Θ = 0:0.1:10
    for θ in Θ
        predicted = predicted_probs .> θ
        tp = sum(predicted .& truth_labels)
        fp = sum(predicted .& .!truth_labels)

        tn = sum(.!predicted .& .!truth_labels)
        fn = sum(.!predicted .& truth_labels)

        append!(tprs, (tp / (tp + fn)))
        append!(fprs, (fp / (fp + tn)))
    end
    return tprs, fprs
end

function compute_aocroc(tprs, fprs)
    aucroc = 0
    for i in eachindex(tprs[2:end])
        aucroc += 0.5 * abs(fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i])
    end
    return aucroc
end

tprs, fprs = roc_curve(truth_labels, predicted_probs)
compute_aocroc(tprs, fprs)