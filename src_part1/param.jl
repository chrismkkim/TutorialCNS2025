dt=0.1;
Ncells=100;

tau=5; #in ms
nloop=50;
learn_every=5.0

# connectivity
K = 100;
g=1.0; #network gain
pconn=0.3

# training param
penlambda = 0.5
train_duration = 100.;
stim_on         = 100.;
stim_off        = 110.;
train_time      = stim_off + train_duration;

Nsteps = train_time/dt

mutable struct paramType
    dt::Float64
    Ncells::Int64
    tau::Float64
    nloop::Int64
    learn_every::Float64
    K::Int64
    g::Float64
    pconn::Float64
    penlambda::Float64
    train_duration::Float64
    stim_on::Int64
    stim_off::Int64
    train_time::Float64
    Nsteps::Int64
    
end

p = paramType(dt,Ncells,tau,nloop,learn_every,K,g,pconn,penlambda,train_duration,stim_on,stim_off,train_time,Nsteps);

