function propagate_spikes_static_rec(ci,nc0,w0Index,w0Weights,forwardInputsE,forwardInputsI)
    for j = 1:nc0[ci]
        cell = w0Index[j,ci]
        wgt  = w0Weights[j,ci]
        if wgt > 0  #E synapse
            forwardInputsE[cell] += wgt
        elseif wgt < 0  #I synapse
            forwardInputsI[cell] += wgt
        end
    end
    return forwardInputsE, forwardInputsI
end

function propagate_spikes_plastic_rec(ci,ncpOut,wpIndexOut,wpWeightOut,forwardInputsP)
    for j = 1:ncpOut[ci]
        cell = Int(wpIndexOut[j,ci])
        forwardInputsP[cell] += wpWeightOut[j,ci]
    end
    return forwardInputsP
end

function propagate_spikes_plastic_ffwd(ci,Ne,wpWeightFfwd,forwardInputsP,licki)
    for j = 1:Ne
        forwardInputsP[j] += wpWeightFfwd[licki][j,ci]
    end
    return forwardInputsP
end


function rls_einet(p, almOrd, matchedCells, r, s, Px, P, 
    wpIndexIn, wpWeightIn, wpWeightOut, wpIndexConvert, wpWeightFfwd, 
    ncpIn, mu, xtarg, learn_seq, synInputBalanced, numExcInh, licki)

    for nid = 1:length(almOrd)
        ci = matchedCells[nid] # model neuron
        ci_alm = almOrd[nid] # alm neuron

        rtrim = @view r[Px[ci]]                 
        raug = [rtrim; s]
        k = P[ci]*raug
        vPv = raug'*k
        den = 1.0/(1.0 + vPv[1])
        BLAS.gemm!('N','T',-den,k,k,1.0,P[ci])                   
        e  = wpWeightIn[ci,:]'*rtrim + wpWeightFfwd[licki][ci,:]'*s + synInputBalanced[ci] + mu[ci] - xtarg[licki][learn_seq,ci_alm]
        dw = -e*k*den
        wpWeightIn[ci,:] .+= dw[1 : numExcInh]
        wpWeightFfwd[licki][ci,:] .+= dw[numExcInh+1 : end]

    end                
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    learn_seq += 1

    return wpWeightIn, wpWeightOut, wpWeightFfwd, learn_seq
end

function genPmatrix(p, wpIndexIn)

    numFfwd = p.Lffwd
    numExc = Int(p.Lexc)
    numInh = Int(p.Linh)
    numExcInh = numExc + numInh
    P = Vector{Array{Float64,2}}(); 
    Px = Vector{Array{Int64,1}}();
    # train a subset of excitatory neurons
    for ci=1:Int(p.Ne)    
        # neurons presynaptic to ci                
        push!(Px, wpIndexIn[ci,:])         
        # ----- Pinv: recurrent -----#
        # row sum penalty
        vec10 = [ones(numExc); zeros(numInh)];
        vec01 = [zeros(numExc); ones(numInh)];
        Pinv_rowsum = p.penmu*(vec10*vec10' + vec01*vec01')
        # L2-penalty
        Pinv_L2 = p.penlamEE*one(zeros(numExcInh,numExcInh))
        # Pinv: recurrent - L2 + Rowsum
        Pinv_rec = Pinv_L2 + Pinv_rowsum
        # ----- Pinv: ffwd - L2 -----#
        Pinv_ffwd = p.penlamFF*one(zeros(numFfwd,numFfwd))
        # ----- Pinv: total -----#
        Pinv = zeros(numExcInh+numFfwd, numExcInh+numFfwd)
        Pinv[1:numExcInh, 1:numExcInh] = Pinv_rec
        Pinv[numExcInh+1 : numExcInh+numFfwd, numExcInh+1 : numExcInh+numFfwd] = Pinv_ffwd
        push!(P, Pinv\one(zeros(numExcInh+numFfwd, numExcInh+numFfwd)))
    end
    return P, Px

end

function funMovAvg(x,wid)
    Nsteps = size(x)[1]
    movavg = zeros(size(x))
    for i = 1:Nsteps
        Lind = maximum([i-wid, 1])
        Rind = minimum([i+wid, Nsteps])
        xslice = @view x[Lind:Rind,:]
        movavg[i,:] = mean(xslice, dims=1)[:]
    end    
    return movavg
end

function funRollingAvg(p,t,wid,widInc,learn_nsteps,movavg,cnt,x,ci)
    startInd = Int(floor((t - p.stim_off - wid)/p.learn_every) + 1)
    endInd = Int(minimum([startInd + widInc, learn_nsteps]))
    if startInd > 0
        movavg[startInd:endInd] .+= x
        if ci == 1
            cnt[startInd:endInd] .+= 1
        end
    else
        movavg[1:endInd] .+= x
        if ci == 1
            cnt[1:endInd] .+= 1
        end
    end
    return movavg, cnt
end

function rls(p, Ncells, r, Px, P, w0WeightIn, w0WeightOut, w0IndexConvert, nc0, mu, xtarg, learn_seq)

    for ctrn = 1:Ncells
        rtrim = @view r[Px[ctrn]]
        k = P[ctrn]*rtrim
        vPv = rtrim'*k
        den = 1.0/(1.0 + vPv[1])
        BLAS.gemm!('N','T',-den,k,k,1.0,P[ctrn])
        e  = w0WeightIn[ctrn,:]'*rtrim + mu[ctrn] - xtarg[learn_seq,ctrn] + 2.0*randn()
        dw = -e*k*den
        w0WeightIn[ctrn,:] .+= dw
    end                
    w0WeightOut = convertWgtIn2Out(p,nc0,w0IndexIn,w0IndexConvert,w0WeightIn,w0WeightOut)
    learn_seq += 1

    return w0WeightIn, w0WeightOut, learn_seq
end

