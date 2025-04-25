function spk2rate(p, matchedCells, times, ns)

    wid = 20 # 20ms
    Nsteps = Int(p.train_time / wid) + 1
    Ntrained = length(matchedCells)
    rates = zeros(Nsteps, Ntrained)

    for ci = 1:Ntrained
        nid = matchedCells[ci]
        nspk = ns[nid]
        if nspk > 0
            for spk = 1:nspk
                spktime = times[nid,spk]
                kk = Int(floor(spktime/wid)) + 1
                rates[kk,ci] += 1 / (wid/1000)
            end
        end
    end

    return rates

end