function lifnet_sim(p,w0Index,w0Weights,nc0)

    # parameters
    train_time          = copy(p.train_time)    
    dt                  = copy(p.dt)
    Nsteps              = copy(p.Nsteps) # network param
    Ncells              = copy(p.Ncells)
    Ne                  = copy(p.Ne)
    Ni                  = copy(p.Ni)
    taue                = copy(p.taue) # neuron param
    taui                = copy(p.taui)
    threshe             = copy(p.threshe)
    vre                 = copy(p.vre)
    muemax              = copy(p.muemax)
    muimax              = copy(p.muimax)
    tauedecay           = copy(p.tauedecay) # synaptic time
    tauidecay           = copy(p.tauidecay)    
    mu                  = zeros(Ncells)
    mu[1:Ne]            .= muemax
    mu[(Ne+1):Ncells]   .= muimax
    thresh              = threshe * ones(Ncells)    
    tau                 = zeros(Ncells)
    tau[1:Ne]          .= taue
    tau[(1+Ne):Ncells] .= taui    

    # activity variables
    times               = [Float64[] for _ in 1:Ncells]
    ns                  = zeros(Int,Ncells)    
    forwardInputsE      = zeros(Ncells) #summed weight of incoming E spikes
    forwardInputsI      = zeros(Ncells)
    forwardInputsEPrev  = zeros(Ncells) #as above, for previous timestep
    forwardInputsIPrev  = zeros(Ncells)    
    xedecay             = zeros(Ncells)
    xidecay             = zeros(Ncells)    
    v                   = threshe*rand(Ncells) #membrane voltage 
    
    # auxiliary variables
    Nexam               = 10
    synExc              = zeros(Nexam,Nsteps)
    synInh              = zeros(Nexam,Nsteps)
    uavg                = zeros(Ncells) #changed from zeros(Nexam)
        
    for ti=1:Nsteps
        if mod(ti,Nsteps/100) == 1  #print percent complete
            print("\r",round(Int,100*ti/Nsteps))
        end
        t = dt*ti;
        forwardInputsE .= 0.0;
        forwardInputsI .= 0.0;
        for ci = 1:Ncells
            xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
            xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
            synInput = xedecay[ci] + xidecay[ci]
                                    
            v[ci] += dt*((1/tau[ci])*(mu[ci]-v[ci] + synInput))
            if v[ci] > thresh[ci]  #spike occurred
                v[ci] = vre
                push!(times[ci],t)
                ns[ci] = ns[ci]+1
                forwardInputsE, forwardInputsI = propagate_spikes_static_rec(ci,nc0,w0Index,w0Weights,forwardInputsE,forwardInputsI)
            end #end if(spike occurred)
            
            # save average synaptic inputs
            uavg[ci] += (synInput + mu[ci]) / Nsteps 
            # save synaptic activity to show the balanced state
            if ci <= Nexam
                synExc[ci,ti] = xedecay[ci] + mu[ci]
                synInh[ci,ti] = xidecay[ci]
            end
        end #end loop over neurons
    
        forwardInputsEPrev = copy(forwardInputsE)
        forwardInputsIPrev = copy(forwardInputsI)
    
    end #end loop over time
    print("\r")
    println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/train_time)," Hz")
    println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/train_time)," Hz")
        
    return times, ns, uavg, synExc, synInh
    
end
    