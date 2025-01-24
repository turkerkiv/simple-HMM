import librosa
import numpy as np
from hmmlearn import hmm
from pydub import AudioSegment


# Step 1: Extract MFCCs
def extract_mfcc(audio_path, n_mfcc=13):
    print(f"\nExtracting MFCCs from: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    print(
        f"Audio loaded successfully! Duration: {len(y)/sr:.2f} seconds, Sample Rate: {sr} Hz"
    )
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    print(f"MFCCs extracted. Shape: {mfccs.shape} (n_mfcc x time_steps)")
    return np.transpose(mfccs)


# Step 2: Train GMM-HMM
"""
About n_states of GMM-HMM

Too Few States:

If n_states is too small, the model may not capture the complexity of the word.

For example, if n_states=1, the entire word is modeled as a single state, which is too simplistic.

Too Many States:

If n_states is too large, the model may overfit the training data.

For example, if n_states=10 for a short word, the model might try to split the word into too many unnecessary parts.

Optimal Number of States:

The optimal value of n_states depends on the complexity of the word and the variability in the data.

For simple words like "evet" or "hayır," n_states=3 or n_states=4 is often a good starting point.
"""


def train_gmm_hmm(mfccs, n_states=4):
    print(f"\nTraining GMM-HMM with {n_states} states...")
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(mfccs)
    print("Training completed!")
    return model


# Step 3: Predict the word
def predict_word(audio_path, models):
    print(f"\nPredicting word for: {audio_path}")
    mfccs = extract_mfcc(audio_path)
    print("Calculating log-likelihood scores for each model...")
    scores = {word: model.score(mfccs) for word, model in models.items()}
    print("Log-likelihood scores:")
    for word, score in scores.items():
        print(f"  {word}: {score:.2f}")
    predicted_word = max(scores, key=scores.get)
    print(f"Predicted word: {predicted_word}")
    return predicted_word


# Main script
if __name__ == "__main__":
    # List of words to train
    words = ["evet", "hayır", "dur", "git"]
    models = {}

    # Step 4: Train GMM-HMM for each word
    print("\n=== Training GMM-HMM Models ===")
    for word in words:
        print(f"\nTraining model for word: {word}")
        mfccs = extract_mfcc(f"4-word-gmmhmm-project/dataset/{word}.wav")
        models[word] = train_gmm_hmm(mfccs)
        print(f"Model for '{word}' trained successfully!")

    # Step 5: Test on a new audio file
    print("\n=== Testing the Model ===")
    test_audio = "4-word-gmmhmm-project/dataset/test_git.wav"
    print(f"Testing on audio file: {test_audio}")
    predicted_word = predict_word(test_audio, models)
    print(f"\nFinal Prediction: The word is '{predicted_word}'")
