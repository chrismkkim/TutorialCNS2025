function genPlasticWeights(p, w0Index, nc0, ns0, matchedCells)

    # rearrange initial weights
    w0 = Dict{Int,Array{Int,1}}()
    for i = 1:p.Ncells
        w0[i] = []
    end
    for preCell = 1:p.Ncells
        for i = 1:nc0[preCell]
            postCell = w0Index[i,preCell]
            push!(w0[postCell],preCell)
        end
    end

    exc_selected = collect(1:p.Ne) #MODIFY
    inh_selected = collect(p.Ne+1:p.Ncells) #MODIFY 
    
    # define weights_plastic    
    wpWeightIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    ncpIn = zeros(Int,p.Ncells)

    
    # trained exc neurons form a cluster
    for ii = 1:length(matchedCells)
        # neuron to be trained
        postCell = matchedCells[ii]

        matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
        indE = sort(shuffle(matchedCells_noautapse)[1:p.L])
        indI = sort(shuffle(inh_selected)[1:p.L])

        # updated wpIndexIn for postcell in matchedCells
        ind = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)

        # (1) update plastic weights to postcell in matchedCells
        # (2) other plastic weights = 0 
        if postCell <= p.Ne
            wpee = p.wpee*ones(p.Lexc)
            wpei = p.wpei*ones(p.Linh)
            wpWeightIn[postCell,:] = [wpee; wpei]
        else
            wpie = p.wpie*ones(p.Lexc)
            wpii = p.wpii*ones(p.Linh)
            wpWeightIn[postCell,:] = [wpie; wpii]
        end
    end
    
    # define feedforward weights to excitatory neurons
    wpWeightFfwd = Vector{Array{Float64,2}}(); 
    for licki = 1:2
        wtmp = randn(p.Ne, p.Lffwd) * p.wpffwd
        push!(wpWeightFfwd, wtmp)
    end

    # get indices of postsynaptic cells for each presynaptic cell
    wpIndexConvert = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexOutD = Dict{Int,Array{Int,1}}()
    ncpOut = zeros(Int,p.Ncells)
    for i = 1:p.Ncells
        wpIndexOutD[i] = []
    end
    for postCell = 1:p.Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[postCell,i]
            push!(wpIndexOutD[preCell],postCell)
            wpIndexConvert[postCell,i] = length(wpIndexOutD[preCell])
        end
    end
    for preCell = 1:p.Ncells
        ncpOut[preCell] = length(wpIndexOutD[preCell])
    end

    # get weight, index of outgoing connections
    ncpOutMax = Int(maximum(ncpOut))
    wpIndexOut = zeros(ncpOutMax,p.Ncells)
    wpWeightOut = zeros(ncpOutMax,p.Ncells)
    for preCell = 1:p.Ncells
        wpIndexOut[1:ncpOut[preCell],preCell] = wpIndexOutD[preCell]
    end
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    
    return wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut
    
end
