import argparse
from database import load_audio_files
from utils import generate_report
import torch
import numpy as np
import librosa

from models.cnn_model import CNNClassifier, audio_to_spectrogram
from models.rnn_lstm_model import RNNLSTMClassifier, audio_to_mfcc
from models.wav2vec_model import Wav2VecClassifier

parser = argparse.ArgumentParser(description="Deepfake Audio Detector CLI")
parser.add_argument('--model', type=str, choices=['cnn', 'rnn', 'wav2vec'], required=True, help="Escolha o modelo")
args = parser.parse_args()

print("Carregando dados...")
X, y = load_audio_files('data/audios')

print(f"Executando modelo: {args.model}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model == 'cnn':
    model = CNNClassifier().to(device)
    X_proc = [audio_to_spectrogram(x) for x in X]
    X_proc = [torch.tensor(x).unsqueeze(0).unsqueeze(0) for x in X_proc]  # [batch, channels, H, W]

elif args.model == 'rnn':
    model = RNNLSTMClassifier().to(device)
    X_proc = [audio_to_mfcc(x) for x in X]
    X_proc = [torch.tensor(x).unsqueeze(0) for x in X_proc]  # [batch, time_steps, features]

elif args.model == 'wav2vec':
    model = Wav2VecClassifier().to(device)
    X_proc = [torch.tensor(x).unsqueeze(0) for x in X]  # [batch, time_steps]

else:
    raise ValueError("Modelo inválido.")

print("Executando inferência...")
y_pred = []
for xi in X_proc:
    xi = xi.float().to(device)
    output = model(xi)
    y_pred.append(int(output.cpu().detach().numpy() >= 0.5))

generate_report(y, y_pred)