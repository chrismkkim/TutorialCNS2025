using Distributions
using PyCall
using PyPlot
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using JLD

# modifiable parameters
lam_list = [0.5, 1.0]
jj = 2

L_list = [6.0, 12.0]
ll = 1

Lffwd_list = [100, 150, 200, 250, 300, 350, 400]
oo = 5

wpffwd_list = [0.0, 1.0, 2.0]
pp = 1

fracTrained_list = vcat(collect(0.1:0.1:0.7), 1824/2500)
qq = 8 # Ntrained = Npyr


include("param.jl")
include("genWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genffwdRate.jl")
include("genStim.jl")
include("genCellsTrained.jl")
include("runinitial.jl")
include("runtrain.jl")
include("runtest.jl")
include("funMovAvg.jl")
include("funSample.jl")
include("funRollingAvg.jl")
include("runperformance.jl")

dirNetwork = "data_network/"
dirALM = "data_alm/"

#----------- initialization --------------#

# run initial balanced network
w0Index, w0Weights, nc0 = genWeights(p)
uavg, ns0, ustd = runinitial(p,w0Index, w0Weights, nc0)

# select cells to be trained
rtarg_lickright = load(dirALM * "movingrate_Pyr1Hz_lickright.jld", "Pyr")
rtarg_lickleft = load(dirALM * "movingrate_Pyr1Hz_lickleft.jld", "Pyr")

# sample a subset of neurons to be trained
Npyr = size(rtarg_lickleft)[1] # Npyr = 1824
Ntrained = Npyr # Num of trained neurons equals Npyr. More generally, Int(p.Ne * p.fracTrained)    
sampledNeurons = sort(shuffle(collect(1:Npyr))[1:Ntrained])
rtarg_lickright = rtarg_lickright[sampledNeurons,:]
rtarg_lickleft = rtarg_lickleft[sampledNeurons,:]
rtarg_mean = (rtarg_lickright + rtarg_lickleft)/2
almOrd, matchedCells = genCellsTrained(rtarg_mean, ns0)

# select plastic weights to be trained
wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, w0Index, nc0, ns0, matchedCells)

stim = Vector{Array{Float64,2}}()
stim_R = genStim(p)
stim_L = genStim(p)
push!(stim, stim_R)
push!(stim, stim_L)

# load targets
xtarg = Vector{Array{Float64,2}}()
utarg_R = transpose(load(dirALM * "utarg1Hz_lickright.jld", "utarg"))[:,sampledNeurons]
utarg_L = transpose(load(dirALM * "utarg1Hz_lickleft.jld", "utarg"))[:,sampledNeurons]
push!(xtarg, utarg_R)
push!(xtarg, utarg_L)

# generate drive
ffwdRate_mean = 5.0
ffwdRate = Vector{Array{Float64,2}}()
ffwdRate_R = genffwdRate(p, ffwdRate_mean)
ffwdRate_L = genffwdRate(p, ffwdRate_mean)
push!(ffwdRate, ffwdRate_R)
push!(ffwdRate, ffwdRate_L)

#----------- save files --------------#
fname_param = dirNetwork * "p.jld"
fname_w0Index = dirNetwork * "w0Index.jld"
fname_w0Weights = dirNetwork * "w0Weights.jld"
fname_nc0 = dirNetwork * "nc0.jld"
fname_wpIndexIn = dirNetwork * "wpIndexIn.jld"
fname_wpIndexOut = dirNetwork * "wpIndexOut.jld"
fname_wpIndexConvert = dirNetwork * "wpIndexConvert.jld"
fname_ncpIn = dirNetwork * "ncpIn.jld"
fname_ncpOut = dirNetwork * "ncpOut.jld"
fname_stim_R = dirNetwork * "stim_R.jld"
fname_stim_L = dirNetwork * "stim_L.jld"
fname_ffwdRate = dirNetwork * "ffwdRate.jld"

fname_uavg = dirNetwork * "uavg.jld"
fname_almOrd = dirNetwork * "almOrd.jld"
fname_matchedCells = dirNetwork * "matchedCells.jld"

save(fname_param,"p", p)
save(fname_w0Index,"w0Index", w0Index)
save(fname_w0Weights,"w0Weights", w0Weights)
save(fname_nc0,"nc0", nc0)
save(fname_wpIndexIn,"wpIndexIn", wpIndexIn)
save(fname_wpIndexOut,"wpIndexOut", wpIndexOut)
save(fname_wpIndexConvert,"wpIndexConvert", wpIndexConvert)
save(fname_ncpIn,"ncpIn", ncpIn)
save(fname_ncpOut,"ncpOut", ncpOut)
save(fname_stim_R,"stim", stim_R)
save(fname_stim_L,"stim", stim_L)
save(fname_ffwdRate,"ffwdRate", ffwdRate)

save(fname_uavg,"uavg", uavg)
save(fname_almOrd,"almOrd", almOrd)
save(fname_matchedCells,"matchedCells", matchedCells)

#----------- run train --------------#
wpWeightIn, wpWeightOut, wpWeightFfwd = runtrain(dirNetwork,p,w0Index,w0Weights,nc0, stim, xtarg,
wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
almOrd, matchedCells, ffwdRate)


#----------- save files --------------#
fname_wpWeightIn = dirNetwork * "wpWeightIn.jld"
fname_wpWeightOut = dirNetwork * "wpWeightOut.jld"
fname_wpWeightFfwd = dirNetwork * "wpWeightFfwd.jld"

save(fname_wpWeightIn,"wpWeightIn", wpWeightIn)
save(fname_wpWeightOut,"wpWeightOut", wpWeightOut)
save(fname_wpWeightFfwd,"wpWeightFfwd", wpWeightFfwd)
