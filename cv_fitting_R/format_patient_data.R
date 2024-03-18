format_patient_data <- function(patient_df){
  
  num_trials <- length(patient_df$a)
  bandit_index <- 1
  choice_index <- 1
  patient_bandit_sequence <- c(patient_df$displayed_stim)
  patient_choice_sequence <- c(patient_df$chosen_stimulus)
  bandits <- c()
  choices <- c()
  
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
  # initialize all 18 possible stimulus combinations (resembles the actual task set-up)
  combo1 <- subset(Q_stimuli, stimuli_num == 1 | stimuli_num == 5 | stimuli_num == 9)
  combo2 <- subset(Q_stimuli, stimuli_num == 1 | stimuli_num == 6 | stimuli_num == 8)
  combo3 <- subset(Q_stimuli, stimuli_num == 2 | stimuli_num == 4 | stimuli_num == 9)
  combo4 <- subset(Q_stimuli, stimuli_num == 2 | stimuli_num == 6 | stimuli_num == 7)
  combo5 <- subset(Q_stimuli, stimuli_num == 3 | stimuli_num == 4 | stimuli_num == 8)
  combo6 <- subset(Q_stimuli, stimuli_num == 3 | stimuli_num == 5 | stimuli_num == 7)
  
for(i in 1:num_trials){
  bandit_combo <- c()
  stimulus_vector <- c()
  choice <- c()
  
  stim_array <- patient_bandit_sequence[bandit_index]
  stim_array <- strsplit(stim_array,split = ",")
  stim_array <- unlist(stim_array)
  
  left_stimulus <- as.numeric(stim_array[1])
  middle_stimulus <- as.numeric(stim_array[2])
  right_stimulus <- as.numeric(stim_array[3])
  
  if ((left_stimulus == 1 | middle_stimulus == 1 | right_stimulus == 1) & (left_stimulus == 5 | middle_stimulus == 5 | right_stimulus == 5) & (left_stimulus == 9 | middle_stimulus == 9 | right_stimulus == 9)) {
    bandit_combo <- 1 
    stimulus_vector <- c(1,5,9)
  } else if ((left_stimulus == 1 | middle_stimulus == 1 | right_stimulus == 1) & (left_stimulus == 6 | middle_stimulus == 6 | right_stimulus == 6) & (left_stimulus == 8 | middle_stimulus == 8 | right_stimulus == 8)) {
    bandit_combo <- 2
    stimulus_vector <- c(1,6,8)
  } else if ((left_stimulus == 2 | middle_stimulus == 2 | right_stimulus == 2) & (left_stimulus == 4 | middle_stimulus == 4 | right_stimulus == 4) & (left_stimulus == 9 | middle_stimulus == 9 | right_stimulus == 9)) {
    bandit_combo <- 3
    stimulus_vector <- c(2,4,9)
  } else if ((left_stimulus == 2 | middle_stimulus == 2 | right_stimulus == 2) & (left_stimulus == 6 | middle_stimulus == 6 | right_stimulus == 6) & (left_stimulus == 7 | middle_stimulus == 7 | right_stimulus == 7)) {
    bandit_combo <- 4
    stimulus_vector <- c(2,6,7)
  } else if ((left_stimulus == 3 | middle_stimulus == 3 | right_stimulus == 3) & (left_stimulus == 4 | middle_stimulus == 4 | right_stimulus == 4) & (left_stimulus == 8 | middle_stimulus == 8 | right_stimulus == 8)) {
    bandit_combo <- 5
    stimulus_vector <- c(3,4,8)
  } else if ((left_stimulus == 3 | middle_stimulus == 3 | right_stimulus == 3) & (left_stimulus == 5 | middle_stimulus == 5 | right_stimulus == 5) & (left_stimulus == 7 | middle_stimulus == 7 | right_stimulus == 7)) {
    bandit_combo <- 6 
    stimulus_vector <- c(3,5,7)
  }
  
  choice <- patient_choice_sequence[choice_index]
  index_of_choice <- which(choice == stimulus_vector)
  choices <- c(choices,index_of_choice)
  bandits <- c(bandits,bandit_combo)
  
  bandit_index <- bandit_index + 1 
  choice_index <- choice_index + 1
}
  patient_df$translated_bandit_combo <- c(bandits)
  patient_df$translated_choices <- c(choices)
  return(patient_df)
}
