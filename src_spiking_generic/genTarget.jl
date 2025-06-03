function genTarget(p,uavg)

    t      = collect(0:p.learn_every:p.train_duration-p.learn_every)
    nstep  = size(t)[1]    
    period = p.train_duration        
    target = zeros(nstep,p.Ncells)
    A      = 0.5
    for ci in 1:p.Ncells
        target[:,ci] = A * sin.(2*pi*(t/period .+ rand())) .+ uavg[ci]
    end
    return target
end

