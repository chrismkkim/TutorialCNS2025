function genWeights(Ncells,m,sd)

  #Generate a fully connected recurrent weight matrix
  w_rec    = zeros((Ncells,Ncells));

  # recurrent connections
  for i=1:Ncells
    for j=1:Ncells
      d = Normal(m,sd)
      w_rec[i,j] = rand(d)
    end
  end


  # No autapse
  for i=1:Ncells
      w_rec[i,i] = 0.;
  end

  return w_rec

end
