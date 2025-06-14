function lifnet_train(dirData,p,w0Index,w0Weights,nc0, stim, xtarg,
    wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
    almOrd, matchedCells, ffwdRate,
    mode)

# copy param
nloop = copy(p.nloop) # train param
penlamEE = copy(p.penlamEE)
penlamFF = copy(p.penlamFF)
penmu = copy(p.penmu)
learn_every = copy(p.learn_every)
stim_on = copy(p.stim_on)
stim_off = copy(p.stim_off)
train_time = copy(p.train_time)
dt = copy(p.dt) # time param
Nsteps = copy(p.Nsteps) 
Ncells = copy(p.Ncells) # network param
Ne = copy(p.Ne)
Ni = copy(p.Ni)
taue = copy(p.taue) # neuron param
taui = copy(p.taui)
threshe = copy(p.threshe)
vre = copy(p.vre)
muemax = copy(p.muemax)
muimax = copy(p.muimax)
tauedecay = copy(p.tauedecay) # synaptic time
tauidecay = copy(p.tauidecay)
taudecay_plastic = copy(p.taudecay_plastic)

# set up variables
mu = zeros(Ncells)
mu[1:Ne] .= muemax
mu[(Ne+1):Ncells] .= muimax
thresh = threshe * ones(Ncells)
tau = zeros(Ncells)
tau[1:Ne] .= taue
tau[(1+Ne):Ncells] .= taui
times = [Float64[] for _ in 1:Ncells]
ns = zeros(Int,Ncells)
times_ffwd = [Float64[] for _ in 1:p.Lffwd]
ns_ffwd = zeros(Int, p.Lffwd)

forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
forwardInputsI = zeros(Ncells)
forwardInputsP = zeros(Ncells)
forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(Ncells)
forwardInputsPPrev = zeros(Ncells)
forwardSpike = zeros(Ncells)
forwardSpikePrev = zeros(Ncells)
ffwdSpike = zeros(p.Lffwd)
ffwdSpikePrev = zeros(p.Lffwd)

xedecay = zeros(Ncells)
xidecay = zeros(Ncells)
xpdecay = zeros(Ncells) 
synInputBalanced = zeros(Ncells)

v = rand(Ncells) #membrane voltage   
r = zeros(Ncells)
s = zeros(p.Lffwd)
bias = zeros(Ncells)

# For training, use both lick-right and lick-left. 
ntarg = 2

# save data to evaluate performance
learn_nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
xtotal = zeros(learn_nsteps,Ncells)
xebal = zeros(learn_nsteps,Ncells)
xibal = zeros(learn_nsteps,Ncells)
xplastic = zeros(learn_nsteps,Ncells)
xtotalcnt = zeros(learn_nsteps)
xebalcnt = zeros(learn_nsteps)
xibalcnt = zeros(learn_nsteps)
xplasticcnt = zeros(learn_nsteps)
wid = 50
widInc = Int(2*wid/p.learn_every - 1)

example_neurons = 25
vtotal_exccell = zeros(Nsteps,example_neurons)
vtotal_inhcell = zeros(Nsteps,example_neurons)
vebal_exccell = zeros(Nsteps,example_neurons)
vibal_exccell = zeros(Nsteps,example_neurons)
vebal_inhcell = zeros(Nsteps,example_neurons)
vibal_inhcell = zeros(Nsteps,example_neurons)
vplastic_exccell = zeros(Nsteps,example_neurons)
vplastic_inhcell = zeros(Nsteps,example_neurons)

# set up training matrices
numFfwd = p.Lffwd
numExc = Int(p.Lexc)
numInh = Int(p.Linh)
numExcInh = numExc + numInh
P, Px = genPmatrix(p, wpIndexIn)

for iloop =1:nloop
    println("Loop no. ",iloop) 
    start_time = time()
    
    for licki = 1:ntarg
        # reset variables
        ns .= 0
        ns_ffwd .= 0
        xedecay .= 0
        xidecay .= 0
        xpdecay .= 0
        r .= 0
        s .= 0
        v = rand(Ncells)
        learn_seq = 1

        for ti=1:Nsteps
            t = dt*ti;
            forwardInputsE .= 0.0;
            forwardInputsI .= 0.0;
            forwardInputsP .= 0.0;
            forwardSpike .= 0.0;            
            ffwdSpike .= 0.0;
            rndFfwd = rand(p.Lffwd)

            if mode == "train"
                if t > Int(stim_off) && t <= Int(train_time) && mod(t, learn_every) == 0
                    wpWeightIn, wpWeightOut, wpWeightFfwd, learn_seq = rls_einet(p, almOrd, matchedCells, r, s, Px, P, 
                                                wpIndexIn, wpWeightIn, wpWeightOut, wpIndexConvert, wpWeightFfwd, 
                                                ncpIn, mu, xtarg, learn_seq, synInputBalanced, numExcInh, licki)                
                end        
            end

            for ci = 1:Ncells
                xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
                xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
                xpdecay[ci] += -dt*xpdecay[ci]/taudecay_plastic + forwardInputsPPrev[ci]/taudecay_plastic
                synInputBalanced[ci] = xedecay[ci] + xidecay[ci]              
                synInput = synInputBalanced[ci] + xpdecay[ci]
                # if training, compute spike trains filtered by plastic synapses
                r[ci] += -dt*r[ci]/taudecay_plastic + forwardSpikePrev[ci]/taudecay_plastic

                # external input
                if t > Int(stim_on) && t < Int(stim_off) 
                    bias[ci] = mu[ci] + stim[licki][ti-Int(stim_on/dt),ci]
                else
                    bias[ci] = mu[ci]
                end

                # neuron voltage dynamics
                v[ci] += dt*((1/tau[ci])*(bias[ci]-v[ci] + synInput))
                if v[ci] > thresh[ci]  #spike occurred
                    v[ci] = vre
                    forwardSpike[ci] = 1.
                    ns[ci] = ns[ci]+1
                    push!(times[ci],t)
                    forwardInputsE, forwardInputsI = propagate_spikes_static_rec(ci,nc0,w0Index,w0Weights,forwardInputsE,forwardInputsI)
                    forwardInputsP                 = propagate_spikes_plastic_rec(ci,ncpOut,wpIndexOut,wpWeightOut,forwardInputsP)
                end #end if(spike occurred)
     
            end #end loop over neurons

            # External input to trained neurons
            if ti > Int(stim_off/dt)
                for ci = 1:p.Lffwd
                    s[ci] += -dt*s[ci]/taudecay_plastic + ffwdSpikePrev[ci]/taudecay_plastic
                    # Poisson external neuron spiked
                    if rndFfwd[ci] < ffwdRate[licki][ti-Int(stim_off/dt),ci]/(1000/p.dt)
                        ffwdSpike[ci] = 1.
                        ns_ffwd[ci] = ns_ffwd[ci]+1
                        push!(times_ffwd[ci],t)
                        forwardInputsP = propagate_spikes_plastic_ffwd(ci,Ne,wpWeightFfwd,forwardInputsP,licki)
                    end #end if spiked
                end #end loop over ffwd neurons
            end #end ffwd input

            forwardInputsEPrev = copy(forwardInputsE)
            forwardInputsIPrev = copy(forwardInputsI)
            forwardInputsPPrev = copy(forwardInputsP)
            forwardSpikePrev = copy(forwardSpike) # if training, compute spike trains
            ffwdSpikePrev = copy(ffwdSpike) # if training, compute spike trains

        end #end loop over time

    end
    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time)


    if mod(iloop,10) == 0
        performance_R = runperformance(p, w0Index, w0Weights, nc0, 
            wpIndexIn, wpIndexOut, wpWeightIn, wpWeightOut, wpIndexConvert, 
            ncpIn, ncpOut, stim, xtarg, almOrd, matchedCells, 
            ffwdRate, wpWeightFfwd, iloop, dirData)
        writedlm(dirData * "_performance_R_loop$(iloop).txt", performance_R)
    end
    # if mod(iloop,50) == 0
    #     fname_wpWeightIn = dirData * "wpWeightIn_loop$(iloop).jld"
    #     fname_wpWeightOut = dirData * "wpWeightOut_loop$(iloop).jld"
    #     fname_wpWeightFfwd = dirData * "wpWeightFfwd_loop$(iloop).jld"
    #     save(fname_wpWeightIn,"wpWeightIn", wpWeightIn)
    #     save(fname_wpWeightOut,"wpWeightOut", wpWeightOut)
    #     save(fname_wpWeightFfwd,"wpWeightFfwd", wpWeightFfwd)
    # end        

end # end loop over trainings


    return wpWeightIn, wpWeightOut, wpWeightFfwd

end
