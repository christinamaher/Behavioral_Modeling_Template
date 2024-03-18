BIC_ACL_patient_cv <- function(v1,v2,v3,v4,starting_params){
  
  tot_num_trials <- length(v2$a)
  
  starting_alpha <- starting_params[1]
  starting_phi <- starting_params[2]
  
  optim_output <- optimx::optimr(par=c(starting_alpha,starting_phi),fn=loglikelihood_ACL_patient,bs=v1,d=v2,o=v3,t_feature=v4,method = "L-BFGS-B",lower=c(0,0),upper=c(1,Inf),control=list(maxit=10000,ndeps=c(1e-5,2))) # this was 1e-4
  neg_loglikelihood <- c(optim_output[["value"]][1])
  BIC <- 2 * log(tot_num_trials) + 2*neg_loglikelihood # this is manual formula for computing BIC; length(start_nums) is the number of parameters in the model which is 2 in this case.
  BIC <- c(BIC)
  
  fitted_alpha <- c(optim_output[["par"]][1]) # subset the fitted alpha value
  
  fitted_phi <- c(optim_output[["par"]][2]) # this needs to be transformed with invlogit() *!
  
  df <- c(BIC,fitted_alpha,fitted_phi,neg_loglikelihood)
  return(df)
  
}