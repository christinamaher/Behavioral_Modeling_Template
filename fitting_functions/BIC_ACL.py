def BIC_ACL(rep_data):
  
  starting_alpha = np.random.random()
  starting_phi = np.random.uniform(0.51,1)
  
  x0 = [starting_alpha,starting_phi]
  
  bnds = ((0, 1), (0.50, 1)) # sequence of min, max pairs 
  
  optim_output = minimize(loglikelihood_ACL, x0, args=(rep_data),method='L-BFGS-B',bounds=bnds)
  neg_loglikelihood = optim_output.fun
  
  BIC = 2 * np.log(108) + 2*neg_loglikelihood  # 108 is the total number of trials for a given repetition
  return BIC