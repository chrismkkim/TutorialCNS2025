function lifnet_sim(p,w0Index,w0Weights)
    train_time      = copy(p.train_time)    
    dt              = copy(p.dt)
    Nsteps          = Int(train_time / dt) # network param
    Ncells          = copy(p.Ncells)
    nc0             = Ncells - 1
    taum            = copy(p.taum) # neuron param
    thresh          = copy(p.thresh)
    vre             = copy(p.vre)
    muemax          = copy(p.muemax)
    taus            = copy(p.taus) # synaptic time
    mu              = muemax * ones(Ncells)    
    taum            = taum * ones(Ncells)    
    times           = [Float64[] for _ in 1:Ncells]
    ns              = zeros(Int,Ncells)    
    forwardInputs   = zeros(Ncells) #summed weight of incoming E spikes
    forwardInputsPrev = zeros(Ncells) #as above, for previous timestep    
    v               = thresh*rand(Ncells) #membrane voltage 
    u               = zeros(Ncells)    
    uavg            = zeros(Ncells)    
    Nexam           = 10
    usave           = zeros(Nsteps,Nexam)
    vsave           = zeros(Nsteps,Nexam)

    for ti=1:Nsteps
        if (mod(ti,Nsteps/100) == 1)  print("\r",round(Int,100*ti/Nsteps)) end #print percent complete            
        t = dt*ti;
        forwardInputs .= 0.0;
        for ci = 1:Ncells
            # update synaptic activity
            u[ci] += -dt*u[ci]/taus + forwardInputsPrev[ci]/taus
            # update membrane potential
            v[ci] += dt*((1/taum[ci])*(-v[ci] + u[ci] + mu[ci] + 2.0*randn()))
            # save membrane voltage, synaptic drive
            if ci <= Nexam 
                vsave[ti,ci] = v[ci] 
                usave[ti,ci] = u[ci]
            end      
            uavg[ci] += (mu[ci] + u[ci]) / Nsteps
            #spike occurred
            if v[ci] > thresh
                v[ci] = vre
                push!(times[ci],t)
                ns[ci] = ns[ci]+1
                forwardInputs = propagate_spikes(ci,nc0,w0Index,w0Weights,forwardInputs)                            
            end #end if(spike occurred)
        end #end loop over neurons
    
        forwardInputsPrev = copy(forwardInputs)
    
    end #end loop over time
    print("\r")
    
    return times, ns, vsave, usave, uavg
    
end
