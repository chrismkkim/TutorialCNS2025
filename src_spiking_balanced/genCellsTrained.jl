function genCellsTrained(p, targetRate, ns)

    networkMean = ns[1:p.Ne] / (p.train_time / 1000)
    targetMean = mean(targetRate, dims=2)[:]
    networkMean_copy = copy(networkMean)
    Npyr = size(targetMean)[1]

    almOrd = sortperm(targetMean, rev=true)
    matchedCells = zeros(Int, Npyr)
    for ci = 1:Npyr
        cell = almOrd[ci]
        idx = argmin(abs.(targetMean[cell] .- networkMean_copy))
        matchedCells[ci] = idx
        networkMean_copy[idx] = -99.0
    end
    
    targetMean_ord = reverse(targetMean[almOrd])
    networkMean_ord = reverse(networkMean[matchedCells])



    return almOrd, matchedCells
end