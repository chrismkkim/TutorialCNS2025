function runperformance(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,xtarg,almOrd,matchedCells,ffwdRate,wpWeightFfwd,iloop,dirData)

    xtotal, xebal, xibal, xplastic, times, ns, 
    vtotal_exc, vtotal_inh, vebal_exc, vibal_exc, 
    vebal_inh, vibal_inh, vplastic_exc, vplastic_inh = runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim,ffwdRate,wpWeightFfwd)



    fname_xtotal = dirData * "xtotal_$(iloop).jld"
    fname_xebal = dirData * "xebal_$(iloop).jld"
    fname_xibal = dirData * "xibal_$(iloop).jld"
    fname_xplastic = dirData * "xplastic_$(iloop).jld"
    fname_times = dirData * "times_$(iloop).jld"
    fname_ns = dirData * "ns_$(iloop).jld"

    save(fname_xtotal,"xtotal", xtotal)
    save(fname_xebal,"xebal", xebal)
    save(fname_xibal,"xibal", xibal)
    save(fname_xplastic,"xplastic", xplastic)
    save(fname_times,"times", times)
    save(fname_ns,"ns", ns)

    tlen = size(xtotal)[1]
    pcor = zeros(length(almOrd))    
    for nid = 1:length(almOrd)
        ci_alm = almOrd[nid] # alm neuron
        ci = matchedCells[nid] # model neuron

        xtarg_slice = @view xtarg[1:tlen,ci_alm]
        xtotal_slice = @view xtotal[:,ci]
        #println(1,tlen,ci_alm)
        pcor[nid] = cor(xtarg_slice, xtotal_slice)
    end

    pcor_mean = mean(pcor)

    return pcor_mean

end