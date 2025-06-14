function lifnet_test(dirData,p,w0Index,w0Weights,nc0, stim, xtarg,
    wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
    almOrd, matchedCells, ffwdRate)

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
P = Vector{Array{Float64,2}}(); 
Px = Vector{Array{Int64,1}}();
# train a subset of excitatory neurons
for ci=1:Int(Ne)    
    # neurons presynaptic to ci                
    push!(Px, wpIndexIn[ci,:])         
    # ----- Pinv: recurrent -----#
    # row sum penalty
    vec10 = [ones(numExc); zeros(numInh)];
    vec01 = [zeros(numExc); ones(numInh)];
    Pinv_rowsum = penmu*(vec10*vec10' + vec01*vec01')
    # L2-penalty
    Pinv_L2 = penlamEE*one(zeros(numExcInh,numExcInh))
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

    
# set up variables
licki = 1
learn_seq = 1

for ti=1:Nsteps
    t = dt*ti;
    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    forwardInputsP .= 0.0;
    forwardSpike .= 0.0;            
    ffwdSpike .= 0.0;
    rndFfwd = rand(p.Lffwd)

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
    
        # save rolling average for performance analysis
        if t > Int(stim_off) && t <= Int(train_time) && mod(t,1.0) == 0
            xtotal[:,ci], xtotalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xtotal[:,ci],xtotalcnt,synInput,ci)
            # xebal[:,ci], xebalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xebal[:,ci],xebalcnt,xedecay[ci],ci)
            # xibal[:,ci], xibalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xibal[:,ci],xibalcnt,xidecay[ci],ci)
            # xplastic[:,ci], xplasticcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xplastic[:,ci],xplasticcnt,xpdecay[ci],ci)
        end                                
        # save for visualization
        if ci <= example_neurons
            vtotal_exccell[ti,ci] = synInput
            vebal_exccell[ti,ci] = xedecay[ci]
            vibal_exccell[ti,ci] = xidecay[ci]
            vplastic_exccell[ti,ci] = xpdecay[ci]
        elseif ci >= Ncells - example_neurons + 1
            vtotal_inhcell[ti,ci-Ncells+example_neurons] = synInput
            vebal_inhcell[ti,ci-Ncells+example_neurons] = xedecay[ci]
            vibal_inhcell[ti,ci-Ncells+example_neurons] = xidecay[ci]
            vplastic_inhcell[ti,ci-Ncells+example_neurons] = xpdecay[ci]
        end
                
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

for k = 1:learn_nsteps
    xtotal[k,:] = xtotal[k,:]/xtotalcnt[k]
    # xebal[k,:] = xebal[k,:]/xebalcnt[k]
    # xibal[k,:] = xibal[k,:]/xibalcnt[k]
    # xplastic[k,:] = xplastic[k,:]/xplasticcnt[k]
end        

return xtotal, xebal, xibal, xplastic, times, ns, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell

end
