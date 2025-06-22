import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

class Wav2VecClassifier(torch.nn.Module):
    def __init__(self):
        super(Wav2VecClassifier, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.wav2vec(input_values).last_hidden_state
            x = outputs.mean(dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x