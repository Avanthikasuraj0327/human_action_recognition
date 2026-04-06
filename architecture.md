# 🏗️ Project Architecture

## 🔷 Overall Architecture

```mermaid
flowchart LR
    A["Input Video"] --> B["Frame Extraction"]
    B --> C["Single-Frame Pipeline"]
    B --> D["Sequence Pipeline"]
    B --> E["Generative / Reconstruction Pipeline"]

    C --> C1["Middle Frame"]
    C1 --> C2["Resize 64x64 + Normalize"]
    C2 --> C3["CNN / MLP"]
    C3 --> C4["25-Class Prediction"]

    D --> D1["Every 5th Frame"]
    D1 --> D2["Resize 128x128 + ImageNet Normalize"]
    D2 --> D3["ResNet34 Features (512)"]
    D2 --> D4["MobileNetV2 Features (1280)"]
    D3 --> D5["Concatenate → 1792"]
    D4 --> D5
    D5 --> D6["Sequence Length = 30"]
    D6 --> D7["RNN / LSTM / GRU / BiLSTM + Attention"]
    D7 --> D8["25-Class Prediction"]

    E --> E1["GAN"]
    E1 --> E2["Synthetic Frame"]

    E --> E3["Autoencoder"]
    E3 --> E4["Reconstruction + MSE"]
