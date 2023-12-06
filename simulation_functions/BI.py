def BI(Trials,beta,target_feature,block_number):

    import numpy as np
    import pandas as pd 
    import matplotlib as plt
    from scipy.optimize import minimize 
    ######### this is an attentional mechanism

    probability_dic = {'p_shape': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667],
        'p_color': [0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667,0.1666667], 
                'f_shape':["circle","oval","square","circle","oval","square","circle","oval","square"],
                'f_color':["yellow","yellow","yellow","blue","blue","blue","orange","orange","orange"],
                'f_stimuli':["yellow, circle","yellow, oval","yellow, square","blue, circle","blue, oval","blue, square","orange, circle","orange, oval","orange, square"],
                'stimuli_num':[0,1,2,3,4,5,6,7,8]}
    
    probability_df = pd.DataFrame(probability_dic)
    
    bandit_sequence = np.arange(18)
    np.random.shuffle(bandit_sequence) # randomly shuffle the bandit sequence
    
    bandit_number = 0 # starting index for bandit sequence
    
    # initate lists to store values for each trial
    cf_shape = []
    cf_color = []
    a = []
    r = []
    bandit_combo = []
    displayed_stim = []
    p_yellow = []
    p_orange = []
    p_blue = []
    p_square = []
    p_oval = []
    p_circle = []
    trial = []
    
    
    for i in range(Trials):
        trial.append(i) # save trial number for plotting
        # initialize all 18 possible stimulus combinations
        combo1 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 8)]
        combo2 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 7)]
        combo3 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 8)]
        combo4 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 6)] 
        combo5 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 7)]
        combo6 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 6)]
        combo7 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 8)]
        combo8 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 7)]
        combo9 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 8)]
        combo10 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 6)]
        combo11 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 7)]
        combo12 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 6)]
        combo13 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 8)]
        combo14 = probability_df[(probability_df["stimuli_num"] == 0) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 7)]
        combo15 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 8)]
        combo16 = probability_df[(probability_df["stimuli_num"] == 1) | (probability_df["stimuli_num"] == 5) | (probability_df["stimuli_num"] == 6)]
        combo17 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 3) | (probability_df["stimuli_num"] == 7)]
        combo18 = probability_df[(probability_df["stimuli_num"] == 2) | (probability_df["stimuli_num"] == 4) | (probability_df["stimuli_num"] == 6)]
    
        all_stim_combinations = [combo1,combo2,combo3,combo4,combo5,combo6,combo7,combo8,combo9,combo10,combo11,combo12,combo13,combo14,combo15,combo16,combo17,combo18]
    
        # index from pre-assigned bandit sequence
        bandits = bandit_sequence[bandit_number] 
        bandit_combo.append(bandits)
        bandits_df = all_stim_combinations[bandits] # subset the current trial's information 
        
        displayed_stim.append(list(bandits_df["stimuli_num"])) # an array with 3 values corresponding to the 3 stimuli that are shown
    
    # save probability distribution for each feature! 
        yellow = probability_df['p_color'][0] 
        p_yellow.append(yellow)
        blue = probability_df['p_color'][3]
        p_blue.append(blue)
        orange = probability_df['p_color'][6]
        p_orange.append(orange)
        square = probability_df['p_shape'][2]
        p_square.append(square)
        circle = probability_df['p_shape'][0]
        p_circle.append(circle)
        oval = probability_df['p_shape'][1]
        p_oval.append(oval)
        
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
        p = np.exp(beta * V_bandit) / sum(np.exp(beta * V_bandit))
        
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
        elif chosen_feature_color == target_feature:
            mu = 0.80
        else: 
            mu = 0.20
  
        reward = int(np.random.random(1)[0] < mu)
        r.append(reward)
        
        non_chosen_indices = np.where((0,1,2) != choice) # get the index of NON chosen stimuli so their value can be updated too 
        non_chosen_index1 = non_chosen_indices[0][0]
        non_chosen_index2 = non_chosen_indices[0][1]

        # Updating phase
        if reward == 1:
            chosen_shape = 0.80 * list(bandits_df["p_shape"])[choice]
            chosen_color = 0.80 * list(bandits_df["p_color"])[choice]
            
            notchosen_shape1 = 0.20 * list(bandits_df["p_shape"])[non_chosen_index1]
            notchosen_color1 = 0.20 * list(bandits_df["p_color"])[non_chosen_index1]
            
            notchosen_shape2 = 0.20 * list(bandits_df["p_shape"])[non_chosen_index2]
            notchosen_color2 = 0.20 * list(bandits_df["p_color"])[non_chosen_index2]
            
        elif reward == 0:
            chosen_shape = 0.20 * list(bandits_df["p_shape"])[choice]
            chosen_color = 0.20 * list(bandits_df["p_color"])[choice]
            
            notchosen_shape1 = 0.80 * list(bandits_df["p_shape"])[non_chosen_index1]
            notchosen_color1 = 0.80 * list(bandits_df["p_color"])[non_chosen_index1]
            
            notchosen_shape2 = 0.80 * list(bandits_df["p_shape"])[non_chosen_index2]
            notchosen_color2 = 0.80 * list(bandits_df["p_color"])[non_chosen_index2]
        
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
      
        bandit_number = bandit_number + 1 
    
    # store rep info     
    target_feature = [target_feature] * 18
    block = [block_number] * 18 
    
    DF = pd.DataFrame(list(zip(block,trial,target_feature,p_yellow,p_orange,p_blue,p_circle,p_square,p_oval,displayed_stim,a,r)),
               columns =['Block','trial','target_feature','p_yellow','p_orange','p_blue','p_circle','p_square','p_oval','displayed_stim','a','r'])
    return DF
