from preprocessor import Preprocessor
import os

""" if file count is -1 then all files will be loaded or it is file count for per word"""


class dataLoader:
    def __init__(self, folderPath="", preprocessor=Preprocessor(), fileCount=-1):
        self.folderPath = folderPath
        self.data = {}
        self.preprocessor = preprocessor
        self.fileCount = fileCount

    def load(self):
        for word in os.listdir(self.folderPath):  # kelime klasörlerini listele
            counter = 0
            word_path = os.path.join(self.folderPath, word)
            if os.path.isdir(word_path):
                self.data[word] = []
                for dosya in os.listdir(word_path):  # kelime klasörüne gir ve sesleri listele
                    counter += 1
                    if counter > self.fileCount:
                        print(str(len(self.data[word])) + " files loaded for " + word)
                        break

                    dosya_path = os.path.join(word_path, dosya)
                    if dosya.endswith(".wav"):
                        signal, sr = self.preprocessor.load_audio(dosya_path)
                        mfcc_features = self.preprocessor.extract_features(signal)
                        self.data[word].append(mfcc_features)
