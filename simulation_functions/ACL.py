def ACL(Trials, alpha, beta, target_feature, phi_color, phi_shape, block_number):

    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 

    value_dic = {'v_shape': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667],
        'v_color': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667], 
                'f_shape':["circle","oval","square","circle","oval","square","circle","oval","square"],
                'f_color':["yellow","yellow","yellow","blue","blue","blue","orange","orange","orange"],
                'f_stimuli':["yellow, circle","yellow, oval","yellow, square","blue, circle","blue, oval","blue, square","orange, circle","orange, oval","orange, square"],
                'stimuli_num':[0,1,2,3,4,5,6,7,8]}
                
    value_df = pd.DataFrame(value_dic)
        
    bandit_sequence = np.arange(18)
    np.random.shuffle(bandit_sequence) # randomly shuffle the bandit sequence
    
    bandit_number = 0 # starting index for bandit sequence
    
      # initiate empty vectors so you can save this trial-by-trial data 
    cf_shape = [] # chosen feature (stimuli's shape - 4,5,6)
    cf_color = []  # chosen shape (stimuli's color - 1,2,3)
    a = [] # choice
    r = [] #reward
    rpe = []
    bandit_combo = []
    displayed_stim = []
    v_choice = []
    trial = []
    correct = []
    
    for i in range(Trials):
        trial.append(i)
              # initialize all 18 possible stimulus combinations
        combo1 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 8)]
        combo2 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 7)]
        combo3 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 8)]
        combo4 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 6)] 
        combo5 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 7)]
        combo6 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 6)]
        combo7 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 8)]
        combo8 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 7)]
        combo9 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 8)]
        combo10 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 6)]
        combo11 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 7)]
        combo12 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 6)]
        combo13 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 8)]
        combo14 = value_df[(value_df["stimuli_num"] == 0) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 7)]
        combo15 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 8)]
        combo16 = value_df[(value_df["stimuli_num"] == 1) | (value_df["stimuli_num"] == 5) | (value_df["stimuli_num"] == 6)]
        combo17 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 3) | (value_df["stimuli_num"] == 7)]
        combo18 = value_df[(value_df["stimuli_num"] == 2) | (value_df["stimuli_num"] == 4) | (value_df["stimuli_num"] == 6)]
    
        all_stim_combinations = [combo1,combo2,combo3,combo4,combo5,combo6,combo7,combo8,combo9,combo10,combo11,combo12,combo13,combo14,combo15,combo16,combo17,combo18]
        
                # index from pre-assigned bandit sequence
        bandits = bandit_sequence[bandit_number] 
        bandit_combo.append(bandits)
        bandits_df = all_stim_combinations[bandits] # subset the current trial's information 
        
        displayed_stim.append(list(bandits_df["stimuli_num"])) # an array with 3 values corresponding to the 3 stimuli that are shown
        
        # calculate attention-weighted value of stimuli
        V_bandit = (bandits_df["v_shape"] * phi_shape) + (bandits_df["v_color"] * phi_color)
        
        # softmax
        p = np.exp(beta * V_bandit) / sum(np.exp(beta * V_bandit))
        
        # choice 
        eps = -1 * (np.finfo(float).eps)
        y = np.array(np.cumsum(p))
        x = np.array([eps,y[0],y[1],y[2]])
        randomnum = np.random.random(1)[0]
        xxx = np.where(x < np.random.random(1)[0])
        choice = np.argmax(x[xxx])
        a.append(choice)
        
         # store chosen shape and color
        chosen_features = list(bandits_df["f_stimuli"])[choice]
        chosen_feature_shape = list(bandits_df["f_shape"])[choice]
        cf_shape.append(chosen_feature_shape)
        chosen_feature_color = list(bandits_df["f_color"])[choice]
        cf_color.append(chosen_feature_color)   
        
        if chosen_feature_shape == target_feature:
            mu = 0.80
            correct.append(1)
        elif chosen_feature_color == target_feature:
            mu = 0.80
            correct.append(1)
        else: 
            mu = 0.20
            correct.append(0)
        
        reward = int(np.random.random(1)[0] < mu)
        r.append(reward)
        
        V_bandit = list(V_bandit) # reformat so it can be subset
        delta = reward - V_bandit[choice]
        rpe.append(delta)
        v_choice.append(V_bandit[choice])
        
                # Updating phase
        updated_shape_value = list(bandits_df["v_shape"])[choice] + alpha * delta * phi_shape
        updated_color_value = list(bandits_df["v_color"])[choice] + alpha * delta  * phi_color
        
        # Update dataframe 
        cf_shape_df = value_df[(value_df["f_shape"] == chosen_feature_shape)]
        cf_shape_index = list(cf_shape_df["stimuli_num"])
        value_df["v_shape"][cf_shape_index] = updated_shape_value
        
        cf_color_df = value_df[(value_df["f_color"] == chosen_feature_color)]
        cf_color_index = list(cf_color_df["stimuli_num"])
        value_df["v_color"][cf_color_index] = updated_color_value

        bandit_number = bandit_number + 1 
 
    alpha = [alpha] * 18
    target_feature = [target_feature] * 18
    phi_color = [phi_color] * 18
    phi_shape = [phi_shape] * 18 
    block = [block_number] * 18 

    DF = pd.DataFrame(list(zip(block,trial,alpha, target_feature, phi_color, phi_shape, a, r, cf_shape, cf_color, rpe, bandit_combo,displayed_stim,correct)),
               columns =['Block','trial','alpha', 'target_feature','phi_color','phi_shape','a','r','cf_shape','cf_color','rpe','bandit_combo','displayed_stim','correct'])
    
    return DF