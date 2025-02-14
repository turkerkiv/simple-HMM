from dataLoader import dataLoader
from preprocessor import Preprocessor
from HMMTrainer import HMMTrainer

if __name__ == "__main__":
    # veri setini yükle ve mfcc özelliklerini çıkar
    # could be seperated into two different functions
    dl = dataLoader(folderPath="dataset", fileCount=200)
    dl.load()

    # Her kelime için HMM modeli oluştur ve eğit
    ht = HMMTrainer(n_components_multiplier=2, n_iter=100)
    for kelime, mfcc_listesi in dl.data.items():
        ht.train_one_HMM(kelime, mfcc_listesi)
        print(f"{kelime} için HMM modeli eğitildi.")

    # predict
    path = "dataset/devam/devam_ZXKY_YOIGNPJ.wav"
    pp = Preprocessor()
    signal, sr = pp.load_audio(path)
    mfcc_features = pp.extract_features(signal)
    print("Predicted word: " + str(ht.predict(mfcc_features)))