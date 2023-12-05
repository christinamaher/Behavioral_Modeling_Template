def fit_BI(rep_data,beta):
    
    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 

    starting_beta = np.random.uniform(1, 50)
  
    x0 = [starting_beta]
  
    bnds = ((1,50),) 
  
    optim_output = minimize(loglikelihood_BI, x0, args=(rep_data),method='L-BFGS-B',bounds=bnds)
    fitted_beta = optim_output.x[0]
  
    data = pd.DataFrame({'fitted_beta': [fitted_beta], 'simulated_beta':[beta]}) 
    return data