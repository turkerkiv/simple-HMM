import librosa


class Preprocessor:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def load_audio(self, path):
        signal, sr = librosa.load(path, sr=self.sr)
        print(
            f"Audio loaded successfully! Duration: {len(signal)/sr:.2f} seconds, Sample Rate: {sr} Hz"
        )
        return signal, sr

    def extract_features(self, signal):
        mfcc_feat = librosa.feature.mfcc(
            y=signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
        )
        features = mfcc_feat.transpose()
        return features


# def load_audio(path, sr=16000):
#     """Ses dosyasını yükle ve önişleme yap"""
#     signal, sr = librosa.load(path, sr=sr)
#     # signal = normalize_audio(signal)
#     # signal = remove_silence(signal)
#     print(
#         f"Audio loaded successfully! Duration: {len(signal)/sr:.2f} seconds, Sample Rate: {sr} Hz"
#     )
#     return signal, sr


# def normalize_audio(signal):
#     """Sinyali normalize et (-1 ile 1 arası)"""
#     return signal / np.max(np.abs(signal) + 1e-5)

# def remove_silence(signal, top_db=25):
#     """Sessizlik kısımlarını kaldır"""
#     trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
#     return trimmed


# def extract_features(signal, sr, n_mfcc=13, add_delta=True):
#     """MFCC + Delta + Delta-Delta özelliklerini çıkar"""

#     mfcc_feat = librosa.feature.mfcc(
#         y=signal,
#         sr=sr,
#         n_mfcc=n_mfcc,
#     )

# if add_delta:
#     delta1 = delta(mfcc_feat, 2)
#     delta2 = delta(delta1, 2)
#     features = np.hstack((mfcc_feat, delta1, delta2))
# else:

#     features = mfcc_feat.transpose()
# features = apply_cmvn(features)
#     return features


# def apply_cmvn(features):
#     """Cepstral Mean and Variance Normalization"""
#     mean = np.mean(features, axis=0)
#     std = np.std(features, axis=0)
#     return (features - mean) / (std + 1e-5)
