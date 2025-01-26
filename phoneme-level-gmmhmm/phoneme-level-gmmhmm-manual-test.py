from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import numpy as np
from hmmlearn import hmm


# Convert and trim audio files (run only once)
def convert_audio(input_path, output_path, sample_rate=16000):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)  # Convert to 16 kHz mono
    audio.export(output_path, format="wav")


def trim_silence(input_path, output_path, silence_thresh=-50):  # Less aggressive
    audio = AudioSegment.from_file(input_path)
    nonsilent_ranges = detect_nonsilent(audio, silence_thresh=silence_thresh)
    start = nonsilent_ranges[0][0]
    end = nonsilent_ranges[-1][1]
    trimmed_audio = audio[start:end]
    trimmed_audio.export(output_path, format="wav")


# Preprocess audio files (run only once)
convert_audio(
    "phoneme-level-gmmhmm/dataset/kale_male.wav",
    "phoneme-level-gmmhmm/dataset/kale_male_16khz.wav",
)

trim_silence(
    "phoneme-level-gmmhmm/dataset/kale_male_16khz.wav",
    "phoneme-level-gmmhmm/dataset/kale_male_16khz_trimmed.wav",
)

# Main code
phoneme_mfccs = {}
mfccs = {}


def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16 kHz
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract MFCCs
    return mfcc.T  # Transpose to get (time_steps, n_mfcc)


def split_mfcc_by_phonemes(mfcc, word):
    mfcc_length = mfcc.shape[0]
    word_length = len(word)
    split_len = mfcc_length // word_length
    phonemes = list(word)
    print(f"Splitting '{word}' into phonemes:", phonemes)

    for i, phoneme in enumerate(phonemes):
        start = i * split_len
        end = (i + 1) * split_len if i < word_length - 1 else mfcc_length
        segment = mfcc[start:end]
        if phoneme not in phoneme_mfccs:
            phoneme_mfccs[phoneme] = []
        phoneme_mfccs[phoneme].append(segment)


def train_phoneme_hmm(segments, n_states=3):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    concatenated_segments = np.vstack(segments)  # Combine all segments for training
    model.fit(concatenated_segments)
    return model


# def train_phoneme_hmm(segments, n_states=2):  # Increased states
#     model = hmm.GaussianHMM(
#         n_components=n_states,
#         covariance_type="diag",
#         n_iter=200,  # Increased iterations
#         init_params="",
#     )
#     model.startprob_ = np.ones(n_states) / n_states
#     model.transmat_ = np.ones((n_states, n_states)) / n_states

#     print(f"Initialized HMM for phoneme with {n_states} states.")

#     for i, segment in enumerate(segments):
#         if len(segment) > 2:  # Only use segments with sufficient length
#             print(f"  Training on segment {i + 1}: shape={segment.shape}")
#             try:
#                 model.fit(segment)
#                 print(f"    Training completed for segment {i + 1}.")
#             except Exception as e:
#                 print(f"    Error training on segment {i + 1}: {e}")
#         else:
#             print(f"  Skipping segment {i + 1} (too short): shape={segment.shape}")

#     print(f"Finished training HMM for phoneme.")
#     return model


# Extract MFCCs for each word
mfccs["kale"] = extract_mfcc("phoneme-level-gmmhmm/dataset/kale_16khz_trimmed.wav")
mfccs["mama"] = extract_mfcc("phoneme-level-gmmhmm/dataset/mama_16khz_trimmed.wav")
mfccs["ekmek"] = extract_mfcc("phoneme-level-gmmhmm/dataset/ekmek_16khz.wav")

for word in mfccs:
    print(f"Word '{word}' has shape: {mfccs[word].shape}")

# Split MFCCs by phonemes
for word, mfcc in mfccs.items():
    split_mfcc_by_phonemes(mfcc, word)

# Print the number of segments for each phoneme
for phoneme in phoneme_mfccs:
    print(f"Phoneme '{phoneme}' has {len(phoneme_mfccs[phoneme])} segments:")
    for i, segment in enumerate(phoneme_mfccs[phoneme]):
        print(f"  Segment {i + 1}: {segment.shape}")

# Train HMMs for each phoneme
phoneme_hmms = {}
for phoneme, segments in phoneme_mfccs.items():
    print(f"Training HMM for phoneme '{phoneme}' with {len(segments)} segments...")
    phoneme_hmms[phoneme] = train_phoneme_hmm(segments)

# Test the system
test_mfcc = extract_mfcc("phoneme-level-gmmhmm/dataset/kale_male_16khz_trimmed.wav")
mfcc_length = test_mfcc.shape[0]
word_length = 4
split_len = mfcc_length // word_length
splitted_mfcc = []
for i in range(4):
    start = i * split_len
    end = (i + 1) * split_len if i < word_length - 1 else mfcc_length
    segment = test_mfcc[start:end]
    splitted_mfcc.append(segment)

predicted_word = ""
for i, segment in enumerate(splitted_mfcc):
    print(f"  Segment {i + 1}: {segment.shape}")
    best_phoneme = None
    best_score = -float("inf")
    for phoneme, model in phoneme_hmms.items():
        try:
            score = model.score(segment)  # Compute log-likelihood
            if score > best_score:
                best_score = score
                best_phoneme = phoneme
        except Exception as e:
            print(f"Error scoring segment for phoneme '{phoneme}': {e}")
    predicted_word += best_phoneme

print(f"Predicted word: {predicted_word}")
