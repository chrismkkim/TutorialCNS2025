#matplotlib = pyimport("matplotlib")
Line2D = matplotlib.lines.Line2D

function genCellsTrained(targetRate, ns)

    train_time = 20000
    networkMean = ns[1:p.Ne] / (train_time / 1000)
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
    

    # figure(figsize=(3.5,3))
    # plot(targetMean_ord, c="black", marker="o", mec="none", ms=4, linestyle="", alpha=0.3, label="data")
    # plot(networkMean_ord, c="darkorange", marker="o", mec="none", ms=2, linestyle="", alpha=0.6, label="model")
    # xticks(fontsize=12)
    # yticks(fontsize=12)
    # xlabel("neuron", fontsize=12)
    # ylabel("firing rate (Hz)", fontsize=12)
    # legend_elements = [Line2D([0], [0], color="black", lw=2, label="data"),
    #                 Line2D([0], [0], color="darkorange", lw=2, label="model")]
    # legend(handles=legend_elements, frameon=false, fontsize=12, handlelength=1)                    
    # tight_layout()

    # savefig(dirfig * "alm_matchCells5k_trained.png", dpi=300)



    return almOrd, matchedCells
end