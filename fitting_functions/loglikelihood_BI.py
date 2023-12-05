def loglikelihood_BL(params,rep):

    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 
    
    beta_param = params[0]
    n_blocks = 6
    Trials = 18
    
    choiceProb_Full = []
    Block_Num = 1
    
    for i in range(n_blocks):
      probability_dic = {'p_shape': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667],
        'p_color': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667], 
                'f_shape':["circle","oval","square","circle","oval","square","circle","oval","square"],
                'f_color':["yellow","yellow","yellow","blue","blue","blue","orange","orange","orange"],
                'f_stimuli':["yellow, circle","yellow, oval","yellow, square","blue, circle","blue, oval","blue, square","orange, circle","orange, oval","orange, square"],
                'stimuli_num':[0,1,2,3,4,5,6,7,8]}
    
      probability_df = pd.DataFrame(probability_dic)
      Rep_DF = rep.loc[rep['Block'] == Block_Num]
      bs = Rep_DF["displayed_stim"]
      choice = Rep_DF["a"]
      outcome = Rep_DF["r"]
      tf = Rep_DF["target_feature"][0]
      
      Trial_Num = 0
      choiceProb = []
      
      for i in range(Trials):
    
        b = bs[Trial_Num]
        bandits_df = probability_df[(probability_df["stimuli_num"] == b[0]) | (probability_df["stimuli_num"] == b[1]) | (probability_df["stimuli_num"] == b[2])]
        
        # calculate stimuli values
        V_bandit = np.array([None,None,None])
        num_bandits = 3
        bandit_index = 0
        for i in np.arange(num_bandits):
                shape_prob = np.array([bandits_df["p_shape"]])[0]
                color_prob = np.array([bandits_df["p_color"]])[0]
                posterior_shape = 0.80 * shape_prob[bandit_index] #### you can add an attention weight here
                posterior_color = 0.80 * color_prob[bandit_index]
                stimuli_value = (posterior_shape + posterior_color) + 0.20 * (1 - (shape_prob[bandit_index] + color_prob[bandit_index]))
                V_bandit[bandit_index] = stimuli_value
                bandit_index = bandit_index + 1
        
        # softmax
        V_bandit = pd.Series(V_bandit).astype(float) # np.exp throws an error otherwise
        p = np.exp(beta_param * V_bandit) / sum(np.exp(beta_param * V_bandit))
        
        c = choice[Trial_Num]
        p = np.array([p])
        cp = p[0][c]
        choiceProb.append(cp)
        
        V_bandit = list(V_bandit)
        
        reward = outcome[Trial_Num]
        
        delta = reward - V_bandit[c]
        
        non_chosen_indices = np.where((0,1,2) != c) # get the index of NON chosen stimuli so their value can be updated too 
        non_chosen_index1 = non_chosen_indices[0][0]
        non_chosen_index2 = non_chosen_indices[0][1]
        
        chosen_features = list(bandits_df["f_stimuli"])[c]
        chosen_feature_shape = list(bandits_df["f_shape"])[c]
        chosen_feature_color = list(bandits_df["f_color"])[c]
        
        # Updating phase
        if reward == 1:
            chosen_shape = 0.80 * list(bandits_df["p_shape"])[c]
            chosen_color = 0.80 * list(bandits_df["p_color"])[c]
            
            notchosen_shape1 = 0.20 * list(bandits_df["p_shape"])[non_chosen_index1]
            notchosen_color1 = 0.20 * list(bandits_df["p_color"])[non_chosen_index1]
            
            notchosen_shape2 = 0.20 * list(bandits_df["p_shape"])[non_chosen_index2]
            notchosen_color2 = 0.20 * list(bandits_df["p_color"])[non_chosen_index2]
            
        elif reward == 0:
            chosen_shape = 0.20 * list(bandits_df["p_shape"])[c]
            chosen_color = 0.20 * list(bandits_df["p_color"])[c]
            
            notchosen_shape1 = 0.80 * list(bandits_df["p_shape"])[non_chosen_index1]
            notchosen_color1 = 0.80 * list(bandits_df["p_color"])[non_chosen_index1]
            
            notchosen_shape2 = 0.80 * list(bandits_df["p_shape"])[non_chosen_index2]
            notchosen_color2 = 0.80 * list(bandits_df["p_color"])[non_chosen_index2]
        
        x = chosen_shape
        b = chosen_color
        c = notchosen_shape1
        d = notchosen_color1
        e = notchosen_shape2
        f = notchosen_color2
        
        # dumb way to normalize 
        chosen_shape = x / (x + b + c + d + e + f)
        chosen_color = b / (x + b + c + d + e + f)
        notchosen_shape1 = c / (x + b + c + d + e + f)
        notchosen_color1 = d / (x + b + c + d + e + f)
        notchosen_shape2 = e / (x + b + c + d + e + f)
        notchosen_color2 = f / (x + b + c + d + e + f)
    
        
        # Update dataframe 
        cf_shape_df = probability_df[(probability_df["f_shape"] == chosen_feature_shape)]
        cf_shape_index = list(cf_shape_df["stimuli_num"])
        probability_df["p_shape"][cf_shape_index] = chosen_shape

        cf_color_df = probability_df[(probability_df["f_color"] == chosen_feature_color)]
        cf_color_index = list(cf_color_df["stimuli_num"])
        probability_df["p_color"][cf_color_index] = chosen_color
        
        ncf_shape1_df = probability_df[(probability_df["f_shape"] == list(bandits_df["f_shape"])[non_chosen_index1]) ]
        ncf_shape1_index = list(ncf_shape1_df["stimuli_num"])
        probability_df["p_shape"][ncf_shape1_index] = notchosen_shape1
        
        ncf_shape2_df = probability_df[(probability_df["f_shape"] == list(bandits_df["f_shape"])[non_chosen_index2]) ]
        ncf_shape2_index = list(ncf_shape2_df["stimuli_num"])
        probability_df["p_shape"][ncf_shape2_index] = notchosen_shape2
        
        ncf_color1_df = probability_df[(probability_df["f_color"] == list(bandits_df["f_color"])[non_chosen_index1]) ]
        ncf_color1_index = list(ncf_color1_df["stimuli_num"])
        probability_df["p_color"][ncf_color1_index] = notchosen_color1
        
        ncf_color2_df = probability_df[(probability_df["f_color"] == list(bandits_df["f_color"])[non_chosen_index2]) ]
        ncf_color2_index = list(ncf_color2_df["stimuli_num"])
        probability_df["p_color"][ncf_color2_index] = notchosen_color2
      
        Trial_Num = Trial_Num + 1
        
      choiceProb_Full.append(choiceProb)
      Block_Num = Block_Num + 1
        
    NegLL = -1 * np.sum(np.log(choiceProb_Full))
    return NegLL