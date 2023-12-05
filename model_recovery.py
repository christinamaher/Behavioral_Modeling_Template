# load relevant packages
import numpy as np
import pandas as pd 
import matplotlib as plt
from scipy.optimize import minimize 

# import relevant functions
from simulation_functions.ACL import ACL
from simulation_functions.BI import BI
from fitting_functions.loglikelihood_ACL import loglikelihood_ACL
from fitting_functions.BIC_ACL import fit_ACL
from fitting_functions.loglikelihood_BI import loglikelihood_BI
from fitting_functions.fit_BI import fit_BI

# Initialize confusion matrix 
confusion_matrix = np.matrix('0 0; 0 0')

n_reps = 100
n_blocks = 6

# MODEL RECOVERY - BL

for i in range(n_reps):
  
    block = 1
    bi_rep_results = []
    feature_list = ["yellow","blue","orange","circle","oval","square"]
    np.random.shuffle(feature_list)
    target_feature_index = 0 
    beta_test = np.random.uniform(1, 50)

    for i in range(n_blocks):
        bi_block_results = []
        tf = feature_list[target_feature_index]
        bi_block_results =  bayesian_inference(Trials=18,beta=beta_test,target_feature=tf,block_number=block)        
        target_feature_index = target_feature_index + 1 # move on to the next tf (6 total)
        block = block + 1
        bi_rep_results.append(bi_block_results) # save results of each block (6 total to a list)
            
    bi_rep_results = pd.concat(bi_rep_results)
    bi_bic = BIC_BI(rep_data=bi_rep_results)
    acl_bic = BIC_ACL(rep_data=bi_rep_results)
    bic_values = [acl_bic,bi_bic] # ACL, BIC
    min_bic_value = min(bic_values)
    lowest_bic_index = np.where(bic_values == min_bic_value)[0][0]

    # update for result of fitting BI reps
    confusion_matrix[0,lowest_bic_index] = confusion_matrix[0,lowest_bic_index] + 1
    
## MODEL RECOVERY - ACL 
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
            
        acl_block_results =  ACL(Trials=18, alpha=alpha_test, beta=13.5, target_feature=tf, phi_color=w_color, phi_shape=w_shape,block_number=block)        
        target_feature_index = target_feature_index + 1 # move on to the next tf (6 total)
        block = block + 1
        acl_rep_results.append(acl_block_results) # save results of each block (6 total to a list)
            
    acl_rep_results = pd.concat(acl_rep_results)
    acl_bic = BIC_ACL(rep_data=acl_rep_results)
    bl_bic = BIC_BI(rep_data=acl_rep_results)
    bic_values = [acl_bic,bl_bic] # ACL, BIC
    min_bic_value = min(bic_values)
    lowest_bic_index = np.where(bic_values == min_bic_value)[0][0]

# update for result of fitting ACL reps
    confusion_matrix[1,lowest_bic_index] = confusion_matrix[1,lowest_bic_index] + 1


confusion_matrix = confusion_matrix / n_reps 

