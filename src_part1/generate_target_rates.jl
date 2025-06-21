function generate_target_rates(p)
    r_targ = zeros(p.Nsteps,p.Ncells)
    t   = collect(1:p.Nsteps)

    for ci=1:p.Ncells
        A = 1.0 
        T1 = 1000
        t1 = T1*rand();
        #t1=T1 #make all sine waves with same phase
      
        r_targ[:,ci] .= A*sin.((t.-t1)*(2*pi/T1)).+1
        
        r_targ[1:Int(p.stim_off/dt),ci] .= 0
    end

    return r_targ

end
