function genWeights(p)
    """
    fully connected network without self-connection (i.e., no autapse)
    """
    # std of synaptic strength
    jstd = 1.0 * p.jee

    # incoming synapses to neurons (used for training)
    nc0 = Int(p.Ncells-1) # outdegree
    w0IndexIn  = zeros(Int,p.Ncells,nc0)
    w0WeightIn = zeros(p.Ncells,nc0)
    for i = 1:p.Ncells
        precells = filter(x->x!=i, collect(1:p.Ncells)) # remove autapse
        w0IndexIn[i,1:nc0] = precells # fixed outdegree nc0
        w0WeightIn[i,:] =  jstd * randn(nc0)
    end

    # get indices of postsynaptic cells for each presynaptic cell
    w0IndexConvert = zeros(p.Ncells, nc0)
    w0IndexOutD = Dict{Int,Array{Int,1}}()
    for i = 1:p.Ncells
        w0IndexOutD[i] = []
    end
    for postCell = 1:p.Ncells
        for i = 1:nc0
            preCell = w0IndexIn[postCell,i]
            push!(w0IndexOutD[preCell],postCell)
            w0IndexConvert[postCell,i] = length(w0IndexOutD[preCell])
        end
    end

    # get weight, index of outgoing connections
    w0IndexOut = zeros(Int,nc0,p.Ncells)
    w0WeightOut = zeros(nc0,p.Ncells)
    for preCell = 1:p.Ncells
        w0IndexOut[:,preCell] = w0IndexOutD[preCell]
    end
    w0WeightOut = convertWgtIn2Out(p,nc0,w0IndexIn,w0IndexConvert,w0WeightIn,w0WeightOut)

    return w0WeightIn, w0WeightOut, w0IndexIn, w0IndexOut, w0IndexConvert


end
