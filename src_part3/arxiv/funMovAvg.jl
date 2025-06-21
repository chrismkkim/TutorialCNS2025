function funMovAvg(x,wid)

Nsteps = size(x)[1]
movavg = zeros(size(x))
for i = 1:Nsteps
    Lind = maximum([i-wid, 1])
    Rind = minimum([i+wid, Nsteps])
    xslice = @view x[Lind:Rind,:]
    movavg[i,:] = mean(xslice, dims=1)[:]
end
    
return movavg

end