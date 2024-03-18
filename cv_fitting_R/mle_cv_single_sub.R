# Fit patient data 
#clean up the global environment 
rm(list=ls())

# load relevant libraries 
library(pracma)
library(tidyverse)
library(NMOF)
library(LaplacesDemon) # useful for reparameterization
library(scales)

# load relevant code 
source("/Users/christinamaher/Documents/Gem_Hunters/data/R_Scripts/RL_Models/Fitting_Functions/format_patient_data.R")

source("/Users/christinamaher/Documents/Gem_Hunters/data/R_Scripts/RL_Models/Fitting_Functions/loglikelihood_UA_patient.R")
source("/Users/christinamaher/Documents/Gem_Hunters/data/R_Scripts/RL_Models/Fitting_Functions/loglikelihood_ACL_patient.R")

source("/Users/christinamaher/Documents/Gem_Hunters/data/R_Scripts/RL_Models/Fitting_Functions/BIC_UA_patient_cv.R")
source("/Users/christinamaher/Documents/Gem_Hunters/data/R_Scripts/RL_Models/Fitting_Functions/BIC_ACL_patient_cv.R")


MS009 <- read.csv("/Users/christinamaher/Documents/Gem_Hunters/data/ieeg/MS009/MS009_clean.csv", header = TRUE)
MS009 <- MS009[(MS009$condition == "hint"),]
MS009_formatted <- format_patient_data(MS009)

formatted_data <- MS009_formatted

# Fit free parameters with patient data
participant_bandit_sequence <-data.frame(formatted_data$translated_bandit_combo,formatted_data$Block)
colnames(participant_bandit_sequence) <- c('bandit_combo','Block')

decision <- data.frame(formatted_data$translated_choices,formatted_data$Block)
colnames(decision) <- c('a','Block')

outcome <- data.frame(formatted_data$r,formatted_data$Block)
colnames(outcome) <- c('r','Block')

tf_sequence <- data.frame(formatted_data$tf,formatted_data$Block)
colnames(tf_sequence) <- c('target_feature','Block')


subject = 'MS009' 

n_iter <- 5 # number of times to fit with a new random starting point for each parameter
n_blocks <- 6 # each participant plays 6 hinted games

# set bounds for free parameter (phi and alpha) in ACL model
lower_bounds <- c(0,0)
upper_bounds <- c(1,Inf)

ua_master <- data.frame()
acl_master <- data.frame()

for (i in 1:n_iter){
  
  ua_test_results <- data.frame()
  acl_test_results <- data.frame()
  
  # set initial conditions
  starting_alpha <- runif(1,0,1)
  starting_phi <- logit(runif(1,0.51,0.99)) # log transform
  
  for (b in 1:n_blocks){
    # determine test and train blocks
    unique_block_nums <- unique(participant_bandit_sequence$Block)
    test_block_index <- unique_block_nums[b]
    train_block_index <- unique_block_nums[which(1:n_blocks != b)]
    
    # subset data by test and train
    train_participant_bandit_sequence <- participant_bandit_sequence[participant_bandit_sequence$Block %in% c(train_block_index), ]
    train_decision <- decision[decision$Block %in% c(train_block_index), ]
    train_outcome <- outcome[outcome$Block %in% c(train_block_index), ]
    train_tf_sequence <- tf_sequence[tf_sequence$Block %in% c(train_block_index), ]
    
    test_participant_bandit_sequence <- participant_bandit_sequence[participant_bandit_sequence$Block %in% c(test_block_index), ]
    test_decision <- decision[decision$Block %in% c(test_block_index), ]
    test_outcome <- outcome[outcome$Block %in% c(test_block_index), ]
    test_tf_sequence <- tf_sequence[tf_sequence$Block %in% c(test_block_index), ]
    
    # fit training data
    BIC1 <- BIC_UA_patient_cv(v1=train_participant_bandit_sequence,v2=train_decision,v3=train_outcome, v4=train_tf_sequence,starting_params = c(starting_alpha))
    ua_train <- c(BIC1[1],BIC1[2],BIC1[3]) # bic, alpha, negLL
    BIC4 <- BIC_ACL_patient_cv(v1=train_participant_bandit_sequence,v2=train_decision,v3=train_outcome, v4=train_tf_sequence,starting_params = c(starting_alpha,starting_phi)) 
    acl_train <- c(BIC4[1],BIC4[2],BIC4[3],BIC4[4]) # bic, alpha, phi, negLL
    
    # test on test data
    ua_test <- loglikelihood_UA_patient(param=c(ua_train[2]), bs=test_participant_bandit_sequence, d=test_decision, o=test_outcome, t_feature=test_tf_sequence)
    acl_test <- loglikelihood_ACL_patient(param=c(acl_train[2],acl_train[3]), bs=test_participant_bandit_sequence, d=test_decision, o=test_outcome, t_feature=test_tf_sequence)
    
    # save testing results
    ua_test_results_temp <- data.frame(ua_test,ua_train[2])
    acl_test_results_temp <- data.frame(acl_test,acl_train[2],invlogit(acl_train[3]))
    ua_test_results <- rbind(ua_test_results, ua_test_results_temp)
    acl_test_results <- rbind(acl_test_results, acl_test_results_temp)
  }
  # average results from all games (data frame should have 6 rows for each test set)
  ua_master <- rbind(ua_master, colMeans(ua_test_results))
  colnames(ua_master) <- c('negLL','learning_rate')
  ua_master$subject <- c(subject)
  acl_master <- rbind(acl_master, colMeans(acl_test_results))
  colnames(acl_master) <- c('negLL','learning_rate','attention_weight')
  acl_master$subject <- c(subject)
}
min_ua <- ua_master[which.min(ua_master$negLL), ]
min_acl <- acl_master[which.min(acl_master$negLL), ]
ifelse(min_ua$negLL < min_acl$negLL, "UA model is a better fit.", "ACL model is a better fit.")