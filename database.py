import os
import librosa
import numpy as np

def load_audio_files(data_dir, sr=16000):
    X = []
    y = []

    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                filepath = os.path.join(folder_path, filename)
                audio, _ = librosa.load(filepath, sr=sr)
                X.append(audio)
                y.append(label)
    return X, y