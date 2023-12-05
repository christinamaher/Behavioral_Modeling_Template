# Behavioral Modeling
Repository for simulating, performing parameter/model recovery, and fitting for two exemplar behavioral models in Python: 
1. **Attention at choice and learning model** =  Rescorla-Wagner variant that implements selective attention to relevant dimensions in a multidimensional state space. This model is based on [Leong & Radulescu et al (2017)]().
2. **Bayesian inference model** = The Bayesian model uses probablistic inference to compute the probability distribution over the identity of the rewarding dimension and feature given all past trials in a multidimensional state space. This model is based on [Wilson & Niv (2012)]().

Simulations are conducted for a multidimensional decision-making task based on [Leong & Radulescu et al. (2017)]() & [Wilson & Niv (2012)](). A maximum likelihood approach based on [Wilson & Collins (2019)]() is used for model/parameter fitting. 

**Model scripts**
- Attention at choice and learning model: `ACL.py`
- Bayesian inference model: `BI.py`

**Simulation scripts**
- `simulation.py`: Script for model simulation.

**Fitting scripts**
- `param_recovery.py`: Script for parameter recovery. 
- `model_recovery.py`: Script for model recovery. 
- `model_fitting.py`: Script for model fitting with experimental data (for example purposes one example simulated dataset will be used. This is available under the following directory `example_data/`
