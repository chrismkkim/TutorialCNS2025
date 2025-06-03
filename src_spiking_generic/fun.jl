function propagate_spikes(ci,nc0,w0Index,w0Weights,forwardInputs)
    for j = 1:nc0
        cell                 = w0Index[j,ci]
        wgt                  = w0Weights[j,ci]
        forwardInputs[cell] += wgt
    end 
    return forwardInputs
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
