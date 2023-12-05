globals().clear() # clears all environment variables

import numpy as np
import pandas as pd 
import matplotlib as plt
from scipy.optimize import minimize 

# Import relevant functions
from simulation_functions.ACL import ACL
from fitting_functions.loglikelihood_ACL import loglikelihood_ACL
from fitting_functions.fit_ACL import fit_ACL

# Simulate and conduct parameter recovery
n_reps = 1000
n_blocks = 6
ACL_DF = []
parameter_recovery_df = []
rep = 1

for i in range(n_reps):
    block = 1
    acl_rep_results = []
    feature_list = ["yellow","blue","orange","circle","oval","square"]
    np.random.shuffle(feature_list)
    target_feature_index = 0 
    alpha_test = np.random.random()
    
    w1 = np.random.uniform(0.51, 1) # attention weight no.1
    w2 = 1 - w1 # attention weight no.2
    attention_weights = [w1,w2]
    attention_weights = sorted(attention_weights) # organize two weights in ascending order

    for i in range(n_blocks):
        acl_block_results = []
        tf = feature_list[target_feature_index]
        
        if tf == 'orange':
            dimension_hint = ['color']
        elif tf == 'blue':
            dimension_hint = ['color']
        elif tf == 'yellow':
            dimension_hint = ['color']
        else: 
            dimension_hint = ['shape']
            
       # allocate attention weights depending on the relevant dimension in a given block
        if dimension_hint[0] == 'color':
            w_shape = attention_weights[0]
            w_color = attention_weights[1]
        elif dimension_hint[0] == 'shape':
            w_shape = attention_weights[1]
            w_color = attention_weights[0]
            
        acl_block_results =  attention_at_choice_and_learning(Trials=18, alpha=alpha_test, beta=13.5, target_feature=tf, phi_color=w_color, phi_shape=w_shape,block_number=block)        
        target_feature_index = target_feature_index + 1 # move on to the next tf (6 total)
        block = block + 1
        acl_rep_results.append(acl_block_results) # save results of each block (6 total to a list)
            
    acl_rep_results = pd.concat(acl_rep_results)
    param_recovery = fit_ACL(rep_data=acl_rep_results,alpha=alpha_test,phi=attention_weights[1])
    parameter_recovery_df.append(param_recovery)

parameter_recovery_df = pd.concat(parameter_recovery_df)

# Visualize results  
import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(parameter_recovery_df['simulated_alpha'], parameter_recovery_df['fitted_alpha'])
plt.xlabel("Simulated alpha")
plt.ylabel("Fitted alpha")
plt.show()


fig2 = plt.figure()
plt.scatter(parameter_recovery_df['simulated_phi'], parameter_recovery_df['fitted_phi'])
plt.xlabel("Simulated phi")
plt.ylabel("Fitted phi")
plt.show()

parameter_recovery_df.corr(method ='pearson')
