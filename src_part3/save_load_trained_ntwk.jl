function save_trained_network(dir_trained_network)

    fname_param     = dir_trained_network * "p.jld"
    fname_w0Index   = dir_trained_network * "w0Index.jld"
    fname_w0Weights   = dir_trained_network * "w0Weights.jld"

    fname_wpWeightIn   = dir_trained_network * "wpWeightIn.jld"
    fname_wpWeightOut  = dir_trained_network * "wpWeightOut.jld"
    fname_wpWeightFfwd = dir_trained_network * "wpWeightFfwd.jld"

    fname_wpIndexIn   = dir_trained_network * "wpIndexIn.jld"
    fname_wpIndexOut   = dir_trained_network * "wpIndexOut.jld"
    fname_wpIndexConvert   = dir_trained_network * "wpIndexConvert.jld"

    fname_nc0   = dir_trained_network * "nc0.jld"
    fname_ncpIn   = dir_trained_network * "ncpIn.jld"
    fname_ncpOut   = dir_trained_network * "ncpOut.jld"

    fname_xtarg   = dir_trained_network * "xtarg.jld"
    fname_stim   = dir_trained_network * "stim.jld"
    fname_almOrd   = dir_trained_network * "almOrd.jld"
    fname_matchedCells   = dir_trained_network * "matchedCells.jld"
    fname_ffwdRate   = dir_trained_network * "ffwdRate.jld"

    save(fname_param, "p", p)
    save(fname_w0Index, "w0Index", w0Index)
    save(fname_w0Weights, "w0Weights", w0Weights)
    save(fname_wpWeightIn, "wpWeightIn", wpWeightIn)
    save(fname_wpWeightOut, "wpWeightOut", wpWeightOut)
    save(fname_wpWeightFfwd, "wpWeightFfwd", wpWeightFfwd)
    save(fname_wpIndexIn, "wpIndexIn", wpIndexIn)
    save(fname_wpIndexOut, "wpIndexOut", wpIndexOut)
    save(fname_wpIndexConvert, "wpIndexConvert", wpIndexConvert)
    save(fname_nc0, "nc0", nc0)
    save(fname_ncpIn, "ncpIn", ncpIn)
    save(fname_ncpOut, "ncpOut", ncpOut)
    save(fname_xtarg, "xtarg", xtarg)
    save(fname_stim, "stim", stim)
    save(fname_almOrd, "almOrd", almOrd)
    save(fname_matchedCells, "matchedCells", matchedCells)
    save(fname_ffwdRate, "ffwdRate", ffwdRate)

end


function load_trained_network(dir_trained_network)

    fname_param = dir_trained_network * "p.jld"
    fname_w0Index = dir_trained_network * "w0Index.jld"
    fname_w0Weights = dir_trained_network * "w0Weights.jld"
    fname_wpWeightIn = dir_trained_network * "wpWeightIn.jld"
    fname_wpWeightOut = dir_trained_network * "wpWeightOut.jld"
    fname_wpWeightFfwd = dir_trained_network * "wpWeightFfwd.jld"
    fname_wpIndexIn = dir_trained_network * "wpIndexIn.jld"
    fname_wpIndexOut = dir_trained_network * "wpIndexOut.jld"
    fname_wpIndexConvert = dir_trained_network * "wpIndexConvert.jld"
    fname_nc0 = dir_trained_network * "nc0.jld"
    fname_ncpIn = dir_trained_network * "ncpIn.jld"
    fname_ncpOut = dir_trained_network * "ncpOut.jld"
    fname_xtarg = dir_trained_network * "xtarg.jld"
    fname_stim = dir_trained_network * "stim.jld"
    fname_almOrd = dir_trained_network * "almOrd.jld"
    fname_matchedCells = dir_trained_network * "matchedCells.jld"
    fname_ffwdRate = dir_trained_network * "ffwdRate.jld"

    
    p = load(fname_param,"p")
    w0Index = load(fname_w0Index,"w0Index")
    w0Weights = load(fname_w0Weights,"w0Weights")
    wpWeightIn = load(fname_wpWeightIn,"wpWeightIn")
    wpWeightOut = load(fname_wpWeightOut,"wpWeightOut")
    wpWeightFfwd = load(fname_wpWeightFfwd,"wpWeightFfwd")
    wpIndexIn = load(fname_wpIndexIn,"wpIndexIn")
    wpIndexOut = load(fname_wpIndexOut,"wpIndexOut")
    wpIndexConvert = load(fname_wpIndexConvert,"wpIndexConvert")
    nc0 = load(fname_nc0,"nc0")
    ncpIn = load(fname_ncpIn,"ncpIn")
    ncpOut = load(fname_ncpOut,"ncpOut")
    xtarg = load(fname_xtarg,"xtarg")
    stim = load(fname_stim,"stim")
    almOrd = load(fname_almOrd, "almOrd")
    matchedCells = load(fname_matchedCells, "matchedCells")
    ffwdRate = load(fname_ffwdRate,"ffwdRate")

    return p, w0Index, w0Weights, wpWeightIn, wpWeightOut, wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, 
        nc0, ncpIn, ncpOut, xtarg, stim, almOrd, matchedCells, ffwdRate
end
        
        