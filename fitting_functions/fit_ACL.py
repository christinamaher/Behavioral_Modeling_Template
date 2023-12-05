def fit_ACL(rep_data,alpha,phi):
    
    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 
  
    starting_alpha = np.random.random()
    starting_phi = np.random.uniform(0.51,1)
  
    x0 = [starting_alpha,starting_phi]
  
    bnds = ((0, 1), (0.50, 1)) # sequence of min, max pairs 
  
    optim_output = minimize(loglikelihood_ACL, x0, args=(rep_data),method='L-BFGS-B',bounds=bnds)
    fitted_alpha = optim_output.x[0]
    fitted_phi = optim_output.x[1]
  
    data = pd.DataFrame({'fitted_alpha': [fitted_alpha], 'fitted_phi':[fitted_phi], 'simulated_alpha':[alpha],'simulated_phi':[phi]}) 
    return data