"""

Theory:
- what HMM is: a statistical model that represents systems with hidden states.
- its components: states, observations, transition probabilities, emission probabilities, and initial probabilities.

- The three main problems HMMs solve:
    - Decoding (finding the most probable sequence of hidden states).
    - Evaluation (calculating the likelihood of a sequence of observations).
    - Learning (estimating parameters from data).

"""

import numpy as np
from hmmlearn import hmm # emission probabilities categorical. Different from GaussianHMM which is normal distribution

# Define the HMM parameters

# STATES: These are the hidden states that we can't see
states = ['Rainy', 'Sunny'] 
n_states = len(states)

# OBSERVATIONS (EMISSIONS): These are the things that we can see.
# They are emitted by the hidden states
# For example if you clean the house, it is more likely that it is rainy outside. 
# So because of being rainy you are cleaning
observations = ['walk', 'shop', 'clean']  
n_observations = len(observations)

# ------------------------------------------------------------------
# ------------- MANUAL DEFINITION OF PARAMETERS --------------------
# ------------------------------------------------------------------

# TRANSITION PROBABILITIES: These are the probabilities of switching between states
# this matrix tells the likelihood of switching between states 
# rainy -> rainy, rainy -> sunny, sunny -> rainy, sunny -> sunny
# square matrix of size n_states x n_states
transition_probabilities = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
]) 

# EMISSION PROBABILITIES: These are the probabilities of seeing observations given the state
# This tells the likelihood of observing a specific activity given a particular hidden state
# clean|rainy = 0.5, shop|rainy = 0.4, walk|rainy = 0.1
# matrix of size n_states x n_observations 
emission_probabilities = np.array([
    [0.1, 0.2, 0.7], # Rainy
    [0.6, 0.3, 0.1] # Sunny
])

# # START PROBABILITIES: These are the probabilities of starting in a particular state
# # This tells the likelihood of starting in a particular hidden state
# # 60% rainy, 40% sunny
start_probabilities = np.array([0.6, 0.4])

# # Create the model
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01, n_trials=1)
model.startprob_ = start_probabilities
model.transmat_ = transition_probabilities
model.emissionprob_ = emission_probabilities

# Simulate a sequence of observations
sequence_length = 10
observed_sequence, states_sequence = model.sample(n_samples=sequence_length)
print("Simulated hidden state: (NOT KNOWN BY PREDICTER)", [states[i] for i in states_sequence])

print("Simulated observed states:", [observations[np.where(observed_sequence[i] == 1)[0][0]] for i in range(len(observed_sequence))])

# Decode hidden states based on observed sequence
predicted_states = model.predict(observed_sequence)
print("Predicted hidden states:", [states[i] for i in predicted_states])

# ------------------------------------------------------------------
# ----------- END - MANUAL DEFINITION OF PARAMETERS ----------------
# ------------------------------------------------------------------

