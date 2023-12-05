def loglikelihood_ACL(params,rep):

    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 
    
    phi_color = []
    phi_shape = []
    alpha_param = params[0]
    phi_param = params[1]
    n_blocks = 6
    Trials = 18
    
    choiceProb_Full = []
    Block_Num = 1
    
    p3 = 1 - phi_param
    
    for i in range(n_blocks):
      value_dic = {'v_shape': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667],
        'v_color': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667], 
                'f_shape':["circle","oval","square","circle","oval","square","circle","oval","square"],
                'f_color':["yellow","yellow","yellow","blue","blue","blue","orange","orange","orange"],
                'f_stimuli':["yellow, circle","yellow, oval","yellow, square","blue, circle","blue, oval","blue, square","orange, circle","orange, oval","orange, square"],
                'stimuli_num':[0,1,2,3,4,5,6,7,8]}
    
      value_df = pd.DataFrame(value_dic)
      Rep_DF = rep.loc[rep['Block'] == Block_Num]
      bs = Rep_DF["displayed_stim"]
      choice = Rep_DF["a"]
      outcome = Rep_DF["r"]
      tf = Rep_DF["target_feature"][0]
      
      if tf == 'orange':
        dimension_hint = ['color']
      elif tf == 'blue':
        dimension_hint = ['color']
      elif tf == 'yellow':
        dimension_hint = ['color']
      else: 
        dimension_hint = ['shape']
        
      if dimension_hint[0] == 'color':
        phi_color = phi_param
        phi_shape = p3
      elif dimension_hint[0] == 'shape':
        phi_shape = phi_param
        phi_color = p3
      
      Trial_Num = 0
      choiceProb = []
      
      for i in range(Trials):
    
        b = bs[Trial_Num]
        bandits_df = value_df[(value_df["stimuli_num"] == b[0]) | (value_df["stimuli_num"] == b[1]) | (value_df["stimuli_num"] == b[2])]
        
        # calculate attention-weighted value of stimuli
        V_bandit = (bandits_df["v_shape"] * phi_shape) + (bandits_df["v_color"] * phi_color)
        
        # softmax
        p = np.exp(13.5 * V_bandit) / sum(np.exp(13.5 * V_bandit)) # softmax inverse temperature is fixed based on Leong and Radulescu (2017) and therefore not being fit 
        p = np.array([p])
        c = choice[Trial_Num]
        cp = p[0][c]
        choiceProb.append(cp)
        
        chosen_features = list(bandits_df["f_stimuli"])[c]
        chosen_feature_shape = list(bandits_df["f_shape"])[c]
        chosen_feature_color = list(bandits_df["f_color"])[c]
        
        V_bandit = list(V_bandit)
        delta = outcome[Trial_Num] - V_bandit[c]
        
        # Updating phase
        updated_shape_value = list(bandits_df["v_shape"])[c] + alpha_param * delta * phi_shape
        updated_color_value = list(bandits_df["v_color"])[c] + alpha_param * delta  * phi_color
        
        # Update dataframe 
        cf_shape_df = value_df[(value_df["f_shape"] == chosen_feature_shape)]
        cf_shape_index = list(cf_shape_df["stimuli_num"])
        value_df["v_shape"][cf_shape_index] = updated_shape_value
        
        cf_color_df = value_df[(value_df["f_color"] == chosen_feature_color)]
        cf_color_index = list(cf_color_df["stimuli_num"])
        value_df["v_color"][cf_color_index] = updated_color_value
            
        Trial_Num = Trial_Num + 1
        
      choiceProb_Full.append(choiceProb)
      Block_Num = Block_Num + 1
        
    NegLL = -1 * np.sum(np.log(choiceProb_Full))
    return NegLL