import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import relevant functions
from simulation_functions.ACL import ACL
from simulation_functions.BI import BI

# Simulate 100 repetitions of Bayesian Inference model (BI) with different free parameter values and visualize results
ACL_DF = []
n_reps = 100
rep = 1 
n_blocks = 6 # 6 blocks x 18 trials each = 108 total trials

for i in range(n_reps):
    acl_rep_results = []
    block = 1
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
    rep_num = [rep] * 108
    acl_rep_results['Rep'] = rep_num
    rep = rep + 1
    ACL_DF.append(acl_rep_results) # list of dataframes


# Calculate accuracy rates by trial and participant
accuracy_by_trial_participant = ACL_DF.groupby(['trial', 'rep'])['correct'].mean().reset_index()

# Calculate mean and SEM across participants for each trial
mean_accuracy_by_trial = accuracy_by_trial_participant.groupby('trial')['correct'].mean()
sem_accuracy_by_trial = accuracy_by_trial_participant.groupby('trial')['correct'].sem()

# Plotting the learning curve with SEM using Matplotlib
plt.figure(figsize=(10, 6))

# Plot mean accuracy
plt.plot(mean_accuracy_by_trial.index, mean_accuracy_by_trial.values, label='Mean Accuracy', marker='o')

# Plot SEM
plt.fill_between(mean_accuracy_by_trial.index,
                 mean_accuracy_by_trial - sem_accuracy_by_trial,
                 mean_accuracy_by_trial + sem_accuracy_by_trial,
                 alpha=0.3, label='SEM')

# Customize the plot
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Simulate 100 repetitions of Bayesian Inference model (BI) with different free parameter values and visualize results
BL_DF = []
n_reps = 100
rep = 1 
n_blocks = 6 # 6 blocks x 18 trials each = 108 total trials

for i in range(n_reps):
    bl_rep_results = []
    block = 1
    feature_list = ["yellow","blue","orange","circle","oval","square"]
    np.random.shuffle(feature_list)
    target_feature_index = 0 
    beta_test = int(np.random.uniform(1, 50))

    for i in range(n_blocks):
        bl_block_results = []
        tf = feature_list[target_feature_index]
        bl_block_results =  BI(Trials=18,beta=beta_test,target_feature=tf,block_number=block)
        bl_rep_results.append(bl_block_results) # save results of each block (6 total to a list)
        target_feature_index = target_feature_index + 1 # move on to the next tf (6 total)
        block = block + 1
    bl_rep_results = pd.concat(bl_rep_results)
    rep_num = [rep] * 108
    bl_rep_results['Rep'] = rep_num
    rep = rep + 1
    BL_DF.append(bl_rep_results) # list of dataframes

# visualize simulation performance 
plt.figure()
subset_df = BL_DF[BL_DF['target_feature'] == 'oval'] # subset block where oval was the target feature and compare learned probability of being target feature across trials as a sanity check (should increase with time). You can try this again with the other shapes (should decrease with time).
accuracy_by_trial_participant = subset_df.groupby(['trial', 'rep'])['p_oval'].mean().reset_index()

# Calculate mean and SEM across participants for each trial
mean_accuracy_by_trial = accuracy_by_trial_participant.groupby('trial')['p_oval'].mean()
sem_accuracy_by_trial = accuracy_by_trial_participant.groupby('trial')['p_oval'].sem()

# Plotting the learning curve with SEM using Matplotlib
plt.figure(figsize=(10, 6))

# Plot mean accuracy
plt.plot(mean_accuracy_by_trial.index, mean_accuracy_by_trial.values, label='Learned probability for true target feature', marker='o')

# Plot SEM
plt.fill_between(mean_accuracy_by_trial.index,
                 mean_accuracy_by_trial - sem_accuracy_by_trial,
                 mean_accuracy_by_trial + sem_accuracy_by_trial,
                 alpha=0.3, label='SEM')

# Customize the plot
plt.xlabel('Trial')
plt.ylabel('Learned probability for true target feature')
plt.legend()
plt.grid(True)
plt.show()