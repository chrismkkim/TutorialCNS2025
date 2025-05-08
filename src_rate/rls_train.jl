function rls_train(ti, p, r, r_targ, P, Px, w_rec)
    
    for ci=1:p.Ncells
                    
        # Extract segment of input timecourse

        rtrim = @view r[ti,Px[ci][:]]
        
        # Update inverse correlation matrix
        
        k = P[ci] * rtrim
        vPv   = rtrim'*k;
        den   = 1.0/(1.0 + vPv[1]);
        P[ci] = P[ci] - k*(k'*den);

        # Compute error 
    
        e  = (w_rec[ci,Px[ci][:]])' * rtrim - r_targ[ti,ci]; 

        # Update recurrent weights

        dw = -e[1]*k*den;
        w_rec[ci,Px[ci][:]] = w_rec[ci,Px[ci][:]] + dw 

    end

    return w_rec, P
end

