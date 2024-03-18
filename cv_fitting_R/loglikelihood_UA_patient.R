loglikelihood_UA_patient <- function(param, bs, d, o, t_feature){

  tot_num_blocks <- length(unique(d$Block))
  block_number <- 1
  block_index <- unique(d$Block)[block_number]
  
  choiceProb_full <- c()
  
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
    
    Q_stimuli$stimuli_num = c(1:9)
    
    #  bandit_sequence <- bs # use the sequence that the agent/participant saw  
    # sample1 <- sample(1:18,18,replace = FALSE) --> use it this way if you want just 18 trials per block wherein each stimulus combination appears only once.
    bandit_number <- 1
    
    # initiate empty vectors so you can save this trial-by-trial data 
    choiceProb <- c()
    
    ## NOW THAT WE'VE CREATED THE ENVIRONMENT 
    for(i in 1:number_trials_current_block){ 
      
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
      
      Q_bandit <- (bandits_df$v_shape * 0.5) + (bandits_df$v_color * 0.5) # this extracts the expected value of the three gems the agent is deciding between 
      
      p <- exp(18.3*Q_bandit) / sum(exp(18.3*Q_bandit)) # compute choice probabilities using a softmax choice rule
      
      choice <-  decision_current_block[bandit_number]
      CP <- p[choice]
      choiceProb <- c(choiceProb,CP)
      
      # get what the choice was in terms of shape and color
      choice_features <- bandits_df$features[choice]
      chosen_feature_shape <- bandits_df$features_shape[choice]
      chosen_feature_color <- bandits_df$features_color[choice]
      
      reward <- outcome_current_block[bandit_number]
      
      delta <- as.numeric(reward) - Q_bandit[choice] # value updating the difference between the obtained reward and the expected conjugate value of the stimulus
      
      # Updating phase 
      bandits_df$v_shape[choice] = bandits_df$v_shape[choice] + param[1] * delta * 0.5  # computing new value for chosen features based on reward outcome; tailored by alpha (learning rate), delta (RPE), and phi shape (attention weighting)
      bandits_df$v_color[choice] = bandits_df$v_color[choice] + param[1] * delta * 0.5 # computing new value for chosen features based on reward outcome; tailored by alpha (learning rate), delta (RPE), and phi color (attention weighting)
      
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
  NegLL <- sum((log(choiceProb_full))) * -1 # compute negative log likelihood (less likely to get rounded to 0)
  return(NegLL)
}

