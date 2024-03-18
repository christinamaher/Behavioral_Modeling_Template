BIC_UA_patient_cv <- function(v1,v2,v3,v4,starting_params){
  
  tot_num_trials <- length(v2$a) #tot_num_blocks * num_trials_per_block # 108
  
  starting_alpha <- starting_params[1]
  
  optim_output <- optimx::optimr(par=c(starting_alpha),fn=loglikelihood_UA_patient,bs=v1,d=v2,o=v3,t_feature=v4,method = c("L-BFGS-B"),lower=c(0),upper=c(1),control=list(maxit=c(10000)))
  neg_loglikelihood <- c(optim_output[["value"]][1])
  
  BIC <- 1 * log(tot_num_trials) + 2*neg_loglikelihood # this is manual formula for computing BIC; length(start_nums) is the number of parameters in the model which is 2 in this case.
  BIC <- c(BIC)
  
  fitted_alpha <- c(optim_output[["par"]][1]) # subset the fitted alpha value
  
  df <- c(BIC,fitted_alpha,neg_loglikelihood)
  return(df)
}