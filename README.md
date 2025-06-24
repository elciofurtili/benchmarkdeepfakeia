
# Projeto de Benchmark de Modelos de IA na Detecção de Deepfakes em Áudio

Este projeto implementa e compara três modelos de inteligência artificial — **CNN**, **RNN-LSTM** e **Wav2Vec 2.0** — aplicados na tarefa de detecção de fraudes em áudios gerados por tecnologias de *deepfake*. O objetivo é avaliar a eficácia de cada arquitetura na identificação de áudios manipulados, considerando diferentes abordagens de modelagem.

Os script foram executados utilizando [ROCm](https://github.com/ROCm/ROCm)

Apesar de parecer que está buscando "CUDA", o PyTorch com ROCm mapeia internamente isso para a GPU AMD corretamente, desde que:
- O PyTorch tenha sido instalado com suporte a ROCm e você esteja com `HIP_VISIBLE_DEVICES` e drivers ROCm corretamente configurados.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
``` 

## 🚀 Estrutura do Projeto

```
/deepfake-audio-benchmark
├── models/                       # Implementação dos modelos
│   ├── cnn_model.py
│   ├── rnn_lstm_model.py
│   └── wav2vec_model.py
├── data/                         # Dados de áudio organizados
│   └── audios/
│       ├── real/
│       └── fake/
├── reports/                      # Relatórios e métricas geradas
│   └── metrics_report.csv
├── utils.py                      # Funções auxiliares
├── database.py                    # Gerenciamento de dados locais
├── main.py                        # Interface CLI para execução
├── requirements.txt               # Dependências
└── README.md                      # Este documento
```

## 🔥 Modelos Implementados

- **CNN**: Convolutional Neural Network aplicada sobre espectrogramas.
- **RNN-LSTM**: Rede recorrente com Long Short-Term Memory para sequências acústicas.
- **Wav2Vec 2.0**: Modelo pré-treinado baseado em Transformers para representação de áudio bruto.

## 📊 Métricas Avaliadas

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **AUC-ROC**

Os relatórios são gerados automaticamente na pasta `/reports/metrics_report.csv`.

## 📦 Dependências

- `torch` (PyTorch)
- `torchaudio`
- `librosa`
- `scikit-learn`
- `pandas`
- `tqdm`
- `transformers`
- `matplotlib`

Instalação recomendada:

```bash
pip install -r requirements.txt
```

## ⚙️ Como Executar

1. Organize seus áudios na pasta `/data/audios/real` e `/data/audios/fake`.
2. Execute via linha de comando:

```bash
python main.py --model cnn
python main.py --model rnn
python main.py --model wav2vec
```

3. Os resultados serão salvos na pasta `/reports`.

## 📚 Datasets Utilizados

|  Dataset  | Tipo de Ataques  | Áudios (Real/Fake) | Língua        | Referência |
|------------|------------------|---------------------|----------------|------------|
| CVoiceFake | Vocoder          | 23.544 / 91.700     | Multilíngue    | [Link](https://dl.acm.org/doi/10.1145/3658644.3670285) |
| MLADDC     | Vocoder          | 80.000 / 160.000    | Multilíngue    | [Link](https://openreview.net/forum?id=ic3HvoOTeU) |
| VSASV      | VC, Replay, Adv. | 164.000 / 174.000   | Multilíngue    | [Link](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html) |

## 📝 Artigo Relacionado

Este projeto é base para o artigo produzido na disciplina de Inteligência Artificial (2025), disponível em: [Artigo PDF](https://github.com/elciofurtili/deepfake-audio-benchmark/blob/main/artigo.pdf)