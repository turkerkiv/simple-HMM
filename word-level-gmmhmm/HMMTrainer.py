from hmmlearn.hmm import GaussianHMM
import numpy as np


class HMMTrainer:
    def __init__(self, n_components_multiplier=1, cov_type="diag", n_iter=1000):
        self.n_components_multiplier = n_components_multiplier
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = {}

    def train_one_HMM(self, word, mfcc_features):
        # Tüm MFCC'leri birleştir
        X = np.vstack(mfcc_features)
        lengths = [len(mfcc) for mfcc in mfcc_features]  # Her segmentin uzunluğu
        # HMM modelini oluştur
        hmm_model = GaussianHMM(
            n_components=len(word) * self.n_components_multiplier,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            verbose=True,
        )
        hmm_model.fit(X, lengths)
        self.models[word] = hmm_model

    def predict(self, features):
        """En yüksek skorlu modeli bul"""
        best_score = -float("inf")
        best_label = None
        for label, model in self.models.items():
            try:
                score = model.score(features)
                print(f"{label} için skor: {score}")
                if score > best_score:
                    best_score = score
                    best_label = label
            except:
                continue
        return best_label
