function lifnet_train(mode,p,w0IndexIn,w0IndexOut,w0WeightIn,w0WeightOut,w0IndexConvert,xtarg,stim)

    train_time = copy(p.train_time)    
    dt = copy(p.dt)
    Nsteps = Int(train_time / dt) # network param
    Ncells = copy(p.Ncells)
    nc0 = Ncells - 1
    Ne = copy(p.Ne)
    Ni = copy(p.Ni)
    taum = copy(p.taum) # neuron param
    thresh = copy(p.thresh)
    vre = copy(p.vre)
    muemax = copy(p.muemax)
    taus = copy(p.taus) # synaptic time
    
    # set up variables
    mu = muemax * ones(Ncells)    
    times = [Float64[] for _ in 1:Ncells]
    ns = zeros(Int,Ncells)
    
    forwardInputs  = zeros(Ncells) #summed weight of incoming E spikes
    forwardSpike   = zeros(Ncells)
    forwardInputsPrev = zeros(Ncells) #as above, for previous timestep
    forwardSpikePrev  = zeros(Ncells)
    
    u = zeros(Ncells)    
    v = thresh*rand(Ncells) #membrane voltage 
    r = zeros(Ncells) 
        
    Nexam = 10
    vsave = zeros(Nsteps,Nexam)
    usave = zeros(Nsteps,Nexam)
    extInput = zeros(Ncells)
    if mode == "train" 
        nloop = p.nloop 
    else
        nloop = 1 
    end
    lam = p.penlambda

    # set up correlation matrix
    P = Vector{Array{Float64,2}}(); 
    Px = Vector{Array{Int64,1}}();
    for ci=1:Ncells
        # neurons presynaptic to ci                
        push!(Px, w0IndexIn[ci,:])             
        # Pinv: recurrent 
        Pinv = lam*one(zeros(Ncells-1,Ncells-1))
        push!(P, Pinv\one(zeros(Ncells-1,Ncells-1)))
    end

    
    for iloop = 1:nloop
        println("Loop no. ",iloop, "\r")         

        # initialize variables
        ns .= 0
        u .= 0
        r .= 0
        v = thresh*rand(Ncells)
        learn_seq = 1

        for ti=1:Nsteps
            if (mod(ti,Nsteps/100) == 1)  print("\r",round(Int,100*ti/Nsteps)) end #print percent complete            
            t = dt*ti;
            forwardInputs .= 0.0;
            forwardSpike .= 0.0;            

            if mode == "train"
                if t > Int(stim_off) && t <= Int(train_time) && mod(t, learn_every) == 0
                    w0WeightIn, w0WeightOut, learn_seq = rls(p, Ncells, r, Px, P, w0WeightIn, w0WeightOut, w0IndexConvert, nc0, mu, xtarg, learn_seq)
                end        
            elseif mode == "test"
                ;
            end

            for ci = 1:Ncells
                # if training, compute synaptically-filtered spike trains
                r[ci] += -dt*r[ci]/taus + forwardSpikePrev[ci]/taus            
                # update synaptic activity
                u[ci] += -dt*u[ci]/taus + forwardInputsPrev[ci]/taus
                # apply stimulus immediately before learning the target patterns
                if t > Int(stim_on) && t < Int(stim_off) 
                    extInput[ci] = mu[ci] + stim[ci]
                else
                    extInput[ci] = mu[ci]
                end
                # update membrane potential
                v[ci] += dt*((1/taum)*(-v[ci] + u[ci] + extInput[ci] + 2.0*randn()))
                # save example neuron's membrane potential
                if ci < Nexam 
                    vsave[ti,ci] = v[ci]
                    usave[ti,ci] = extInput[ci] + u[ci] 
                end
                # spike occurred
                if v[ci] > thresh
                    v[ci] = vre
                    ns[ci] = ns[ci]+1
                    forwardSpike[ci] = 1.
                    push!(times[ci],t)
                    forwardInputs = propagate_spikes(ci,nc0,w0IndexOut,w0WeightOut,forwardInputs)
                end #end if(spike occurred)
            end #end loop over neurons
        
            forwardInputsPrev = copy(forwardInputs)
            forwardSpikePrev  = copy(forwardSpike) # if training, compute spike trains
        
        end #end loop over time
        print("\r")
    end    

    return times, ns, vsave, usave
    
end
