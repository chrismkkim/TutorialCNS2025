function load_files(dirData, lickRL)

    if lickRL == "right"
        fname_stim = dirData * "stim_R.jld"
    elseif lickRL == "left"
        fname_stim = dirData * "stim_L.jld"
    end
    fname_ffwdRate = dirData * "ffwdRate.jld"    
    fname_wpWeightFfwd = dirData * "wpWeightFfwd_loop$(500).jld"

    fname_param = dirData * "p.jld"
    fname_w0Index = dirData * "w0Index.jld"
    fname_w0Weights = dirData * "w0Weights.jld"
    fname_nc0 = dirData * "nc0.jld"
    fname_wpIndexIn = dirData * "wpIndexIn.jld"
    fname_wpIndexOut = dirData * "wpIndexOut.jld"
    fname_wpIndexConvert = dirData * "wpIndexConvert.jld"
    # fname_wpWeightIn = dirData * "wpWeightIn.jld"
    fname_wpWeightIn = dirData * "wpWeightIn_loop$(500).jld"
    # fname_wpWeightOut = dirData * "wpWeightOut.jld"
    fname_wpWeightOut = dirData * "wpWeightOut_loop$(500).jld"
    fname_ncpIn = dirData * "ncpIn.jld"
    fname_ncpOut = dirData * "ncpOut.jld"
    # fname_bias = dirData * "bias.jld"
    # fname_phases = dirData * "phases.jld"
    # fname_uavg = dirData * "uavg.jld"    
    fname_almOrd = dirData * "almOrd.jld"
    fname_matchedCells = dirData * "matchedCells.jld"
    
    
    p = load(fname_param,"p")
    w0Index = load(fname_w0Index,"w0Index")
    w0Weights = load(fname_w0Weights,"w0Weights")
    nc0 = load(fname_nc0,"nc0")
    wpIndexIn = load(fname_wpIndexIn,"wpIndexIn")
    wpIndexOut = load(fname_wpIndexOut,"wpIndexOut")
    wpIndexConvert = load(fname_wpIndexConvert,"wpIndexConvert")
    wpWeightIn = load(fname_wpWeightIn,"wpWeightIn")
    wpWeightOut = load(fname_wpWeightOut,"wpWeightOut")
    ncpIn = load(fname_ncpIn,"ncpIn")
    ncpOut = load(fname_ncpOut,"ncpOut")
    almOrd = load(fname_almOrd, "almOrd")
    matchedCells = load(fname_matchedCells, "matchedCells")
    stim = load(fname_stim,"stim")
    ffwdRate = load(fname_ffwdRate,"ffwdRate")
    wpWeightFfwd = load(fname_wpWeightFfwd,"wpWeightFfwd")

    return p, w0Index, w0Weights, nc0, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, stim, almOrd, matchedCells, ffwdRate, wpWeightFfwd
    
end
        
        