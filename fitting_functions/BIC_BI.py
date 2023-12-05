def BIC_BI(rep_data):
  
  starting_beta = np.random.uniform(1, 50)
  
  x0 = [starting_beta]
  
  bnds = ((1,50),) 
  
  optim_output = minimize(loglikelihood_BI, x0, args=(rep_data),method='L-BFGS-B',bounds=bnds)
  neg_loglikelihood = optim_output.fun
  
  BIC = 2 * np.log(108) + 2*neg_loglikelihood  # 108 is the total number of trial for a given repetition
  return BIC