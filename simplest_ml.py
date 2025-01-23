import numpy as np
from hmmlearn import hmm

# Define the HMM structure
states = ['Rainy', 'Sunny']
n_states = len(states)

observations = ['walk', 'shop', 'clean']
n_observations = len(observations)

# Define the true parameters
transition_probabilities = np.array([
    [0.7, 0.3],  # Rainy -> Rainy, Rainy -> Sunny
    [0.4, 0.6]   # Sunny -> Rainy, Sunny -> Sunny
])

emission_probabilities = np.array([
    [0.1, 0.2, 0.7],  # Rainy: walk (0.1), shop (0.2), clean (0.7)
    [0.6, 0.3, 0.1]   # Sunny: walk (0.6), shop (0.3), clean (0.1)
])

start_probabilities = np.array([0.6, 0.4])  # 60% Rainy, 40% Sunny

# Initialize the true model
true_model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01, n_trials=1)
true_model.startprob_ = start_probabilities
true_model.transmat_ = transition_probabilities
true_model.emissionprob_ = emission_probabilities

# Generate a sequence of length 100,000
sequence_length = 100000
observed_sequence, states_sequence = true_model.sample(sequence_length)

# Print the first 10 observations (decoded)
print("First 10 observations (decoded):")
print([observations[np.argmax(obs)] for obs in observed_sequence[:10]])
print("First 10 true hidden states:")
print([states[i] for i in states_sequence[:10]])

# Initialize the HMM to be trained
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01, n_trials=1, init_params='')
model.transmat_ = np.array([[0.5, 0.5], [0.5, 0.5]])
model.startprob_ = np.array([0.5, 0.5])
model.emissionprob_ = np.array([[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]])

# Function to calculate success rate
def calculate_success_rate(model, observed_sequence, states_sequence):
    predicted_states = model.predict(observed_sequence)
    correct_predictions = np.sum(predicted_states == states_sequence)
    total_predictions = len(states_sequence)
    success_rate = correct_predictions / total_predictions
    return success_rate

# Train the HMM with progress tracking
print("\nTraining the HMM...")
for iteration in range(model.n_iter):
    model.fit(observed_sequence)  # Train for one iteration
    log_likelihood = model.score(observed_sequence)  # Get the log-likelihood
    success_rate = calculate_success_rate(model, observed_sequence, states_sequence)  # Calculate success rate

    # Print progress every 10 iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}:")
        print(f"  Log-Likelihood: {log_likelihood:.2f}")
        print(f"  Success Rate: {success_rate * 100:.2f}%")
        print("  Current Transition Probabilities:")
        print(model.transmat_)
        print("  Current Emission Probabilities:")
        print(model.emissionprob_)
        print("  Current Start Probabilities:")
        print(model.startprob_)
        print("-" * 50)

# Inspect the learned parameters
print("\nLearned Transition Probabilities:")
print(model.transmat_)

print("\nLearned Emission Probabilities:")
print(model.emissionprob_)

print("\nLearned Start Probabilities:")
print(model.startprob_)

# Decode the most likely hidden states for the observed sequence
predicted_states = model.predict(observed_sequence)
print("\nPredicted Hidden States (first 10):")
print([states[i] for i in predicted_states[:10]])

# Calculate the final success rate
success_rate = calculate_success_rate(model, observed_sequence, states_sequence)
print(f"\nFinal Success Rate: {success_rate * 100:.2f}%")