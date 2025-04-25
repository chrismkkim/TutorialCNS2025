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
