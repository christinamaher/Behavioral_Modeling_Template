# load relevant packages
import numpy as np
import pandas as pd 
import matplotlib as plt
from scipy.optimize import minimize 

# import relevant functions
from fitting_functions.loglikelihood_ACL import loglikelihood_ACL
from fitting_functions.BIC_ACL import fit_ACL
from fitting_functions.loglikelihood_BI import loglikelihood_BI
from fitting_functions.BIC_BI import fit_BI

# load sample data
df = pd.read_csv('/Users/christinamaher/Documents/GitHub/Behavioral_Modeling/example_data/data.csv')
bi_bic = BIC_BI(rep_data=df)
acl_bic = BIC_ACL(rep_data=df)

# Compare model performance
if bi_bic < acl_bic:
    print("Bayesian inference is the best fitting model.")
elif acl_bic < bi_bic:
    print("Attention at choice and learning is the best fitting model.")
else:
    print("An error was encountered in model fitting.")