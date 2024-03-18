loglikelihood_ACL_patient <- function(param, bs, d, o, t_feature){
  
  #param=c(MS014$alpha[1],MS014$phi[1])
  #bs=participant_bandit_sequence
  #d=decision
  #o=outcome
  #t_feature=tf_sequence
  
  # if you provide the param values (alpha, phi) you can calculate trial-by-trial values for this patient. (i.e., param=c(0.2748831,0.7310586))
  
  #phi_param <- param[2] # extracting
  phi_param <- invlogit(param[2]) # fitting
  p3 <- 1-phi_param
  
  tot_num_blocks <- length(unique(d$Block))
  block_number <- 1
  block_index <- unique(d$Block)[block_number]
  
  choiceProb_full <- c()
  expected_value <- c()
  rpe <- c()
  
  for (i in 1:tot_num_blocks) {
    
    bs$Block <- as.numeric(bs$Block)
    bs_current_block <- bs %>% filter(Block == block_index)
    bs_current_block <- c(bs_current_block$bandit_combo)
    
    d$Block <- as.numeric(d$Block)
    decision_current_block <- d %>% filter(Block == block_index)
    decision_current_block <- c(decision_current_block$a)
    
    o$Block <- as.numeric(o$Block)
    outcome_current_block <- o %>% filter(Block == block_index)
    outcome_current_block <- c(outcome_current_block$r)
    
    t_feature$Block <- as.numeric(t_feature$Block)
    tf_current_block <- t_feature %>% filter(Block == block_index)
    tf_current_block <- c(tf_current_block$target_feature)
    
    number_trials_current_block <- length(outcome_current_block)
    
    ### THIS FIRST SECTION CREATES A DATAFRAME THAT STORES THE BANDIT OPTIONS AND THEIR VALUE
    Q_shapes = c(0.1666667,0.1666667,0.1666667) # start with a value for each feature (1-6) ---> q is usually the value of an action; V is equal to the value of the stimuli. 
    v_shape = rep(Q_shapes,each=3) # hacky way to line them up and recreate the dataframe commented above
    Q_color = c(0.1666667,0.1666667,0.1666667)
    
    Q_stimuli = as.data.frame(v_shape)
    Q_stimuli$v_color = c(Q_color)
    Q_stimuli$features = c("yellow, circle","yellow, oval","yellow, square",  
                           "blue, circle","blue, oval","blue, square","orange, circle",  
                           "orange, oval","orange, square")  
    
    Q_stimuli$features_color <- c("yellow","yellow","yellow","blue",
                                  "blue","blue","orange","orange","orange")
    
    Q_stimuli$features_shape <-  c("circle","oval","square","circle",
                                   "oval","square","circle","oval","square")
    
    dimension_hint <- ifelse(tf_current_block[1] == 'yellow', 'color',  ### FOR EMU DATA THIS NEEDS TO BE STRING VALUES!!!!!!!!!!!!!!!
                             ifelse(tf_current_block[1] == 'blue', 'color',
                                    ifelse(tf_current_block[1] == 'orange','color', 'shape'))) 
    
    #dimension_hint <- ifelse(tf_current_block[1] == 1, 'color',  ### FOR ONLINE DATA THIS NEEDS TO BE NUMERIC VALUES!!!!!!!!!!!!!!!
    #ifelse(tf_current_block[1] == 2, 'color',
    #ifelse(tf_current_block[1] == 3,'color', 'shape'))) 
    
    ### need to add something here. 
    phi_color <- ifelse(dimension_hint == 'color', phi_param,p3) # sets a hint based on the target feature (p4 is smaller, p3 is larger)
    phi_shape <- ifelse(dimension_hint == 'shape', phi_param,p3) # sets a hint based on the target feature
    
    Q_stimuli$v_stimuli = (Q_stimuli$v_shape * phi_shape) + (Q_stimuli$v_color * phi_color)
    Q_stimuli$stimuli_num = c(1:9)
    
    #  bandit_sequence <- bs # use the sequence that the agent/participant saw  
    # sample1 <- sample(1:18,18,replace = FALSE) --> use it this way if you want just 18 trials per block wherein each stimulus combination appears only once.
    bandit_number <- 1
    
    choiceProb <- c()
    
    
    ## NOW THAT WE'VE CREATED THE ENVIRONMENT 
    for(i in 1:length(decision_current_block)){   #number_trials_current_block -- hard coded method is not flexible in the event that participants miss trials. 
      
      # initialize all 18 possible stimulus combinations (resembles the actual task set-up)
      combo1 <- subset(Q_stimuli, stimuli_num == 1 | stimuli_num == 5 | stimuli_num == 9)
      combo2 <- subset(Q_stimuli, stimuli_num == 1 | stimuli_num == 6 | stimuli_num == 8)
      combo3 <- subset(Q_stimuli, stimuli_num == 2 | stimuli_num == 4 | stimuli_num == 9)
      combo4 <- subset(Q_stimuli, stimuli_num == 2 | stimuli_num == 6 | stimuli_num == 7)
      combo5 <- subset(Q_stimuli, stimuli_num == 3 | stimuli_num == 4 | stimuli_num == 8)
      combo6 <- subset(Q_stimuli, stimuli_num == 3 | stimuli_num == 5 | stimuli_num == 7)

      all_stim_combinations <- list(combo1,combo2,combo3,combo4,combo5,combo6)
      
      # index from the bandit sequence that was pre-assigned. 
      bandits <- bs_current_block[bandit_number] # this basically pulls the three gems that the agent should be deciding between
      bandits_df <- all_stim_combinations[[bandits]] # this subsets the gems that the agent should be deciding between
      
      Q_bandit <- (bandits_df$v_shape * phi_shape) + (bandits_df$v_color * phi_color) # this extracts the expected value of the three gems the agent is deciding between 
      
      p <- exp(13.5*Q_bandit) / sum(exp(13.5*Q_bandit)) # compute choice probabilities using a softmax choice rule
      
      choice <-  decision_current_block[bandit_number]
      CP <- p[choice]
      choiceProb <- c(choiceProb,CP)
      
      # get what the choice was in terms of shape and color
      choice_features <- bandits_df$features[choice]
      chosen_feature_shape <- bandits_df$features_shape[choice]
      chosen_feature_color <- bandits_df$features_color[choice]
      
      reward <- outcome_current_block[bandit_number]
      
      delta <- as.numeric(reward) - Q_bandit[choice] # value updating the difference between the obtained reward and the expected conjugate value of the stimulus
      rpe <- c(rpe,delta) # save to vector
      
     # ev <- ifelse(dimension_hint == 'color', bandits_df$v_color[choice],bandits_df$v_shape[choice])
      ev <- Q_bandit[choice]
      expected_value <- c(expected_value,ev) # save to vector 
      
      # Updating phase 
      bandits_df$v_shape[choice] = bandits_df$v_shape[choice] + param[1] * delta * phi_shape  # computing new value for chosen features based on reward outcome; tailored by alpha (learning rate), delta (RPE), and phi shape (attention weighting)
      bandits_df$v_color[choice] = bandits_df$v_color[choice] + param[1] * delta  * phi_color # computing new value for chosen features based on reward outcome; tailored by alpha (learning rate), delta (RPE), and phi color (attention weighting)
      
      cf_shape_df <- subset(Q_stimuli, features_shape == bandits_df$features_shape[choice])
      cf_shape_index <- cf_shape_df$stimuli_num
      Q_stimuli$v_shape[cf_shape_index] <- bandits_df$v_shape[choice] # update the main DF for the next iteration * ANYWHERE that the chosen shape appears
      
      cf_color_df <- subset(Q_stimuli, features_color == bandits_df$features_color[choice])
      cf_color_index <- cf_color_df$stimuli_num
      Q_stimuli$v_color[cf_color_index] <- bandits_df$v_color[choice] # update the main DF for the next iteration * ANYWHERE that the chosen color appears
      
      bandit_number <- bandit_number + 1 # move on to the next in the row of bandits presented
      # each stimuli has two features so for one choice made = two feature values are updated. 
    }
    block_number <- block_number + 1
    block_index <- unique(d$Block)[block_number]
    choiceProb_full <- c(choiceProb_full,choiceProb)
  }
  NegLL <- sum((log(choiceProb_full))) * -1  # compute negative log likelihood (less likely to get rounded to 0)
  return(NegLL)
}

#### For saving patient estimates ####
#df <- data.frame(expected_value,rpe)
#setwd('/Users/christinamaher/Documents/Gem_Hunters/data/ieeg/MS014/')
#write.csv(df, 'model_based_regressors.csv')



