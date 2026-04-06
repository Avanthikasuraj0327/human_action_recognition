# ActionAI: Multi-Model Human Action Recognition System

## Overview

ActionAI is a deep learning project for human action understanding using multiple model families in one deployed application. The system compares spatial classifiers, temporal sequence models, a generative adversarial network, and an autoencoder inside a single Streamlit interface.

This project includes:

- action classification with `CNN`, `MLP`, `RNN`, `LSTM`, `GRU`, and `BiLSTM + Attention`
- synthetic frame generation with `GAN`
- reconstruction and anomaly-style analysis with `Autoencoder`
- a deployed Streamlit UI for video upload, inference, and model comparison

The strongest action-recognition model in the final application is `BiLSTM + Attention`, while the remaining models are included for comparison, analysis, and experimentation.

## Project Structure

### Deployment Folder

- `app.py`: Streamlit deployment application
- `requirements.txt`: Python dependencies
- `README.md`: project documentation
- `cnn_model.pth`: trained CNN weights
- `mlp_model.pth`: trained MLP weights
- `RNN.keras`: trained vanilla RNN model
- `LSTM.keras`: trained LSTM model
- `GRU.keras`: trained GRU model
- `BiLSTM+Attn.keras`: trained BiLSTM with attention model
- `generator_final.pth`: trained GAN generator
- `discriminator_final.pth`: trained GAN discriminator
- `autoencoder_best.pth`: trained autoencoder

### Training Notebooks

The following notebooks were used to train the models:

- `action-recognition-lstm-rnn-gru (1).ipynb`
- `cnn-mlp-action-recognition-dl.ipynb`
- `gan-ae-action-recognition.ipynb`

## Models Used

| Model | Framework | Role | Final Input Used in App |
|---|---|---|---|
| CNN | PyTorch | spatial classification baseline | middle frame, `3 x 64 x 64` |
| MLP | PyTorch | dense classification baseline | flattened normalized middle frame |
| RNN | Keras | temporal classification baseline | `30 x 1792` sequence |
| LSTM | Keras | gated temporal model | `30 x 1792` sequence |
| GRU | Keras | efficient gated temporal model | `30 x 1792` sequence |
| BiLSTM + Attention | Keras | strongest temporal classifier | `30 x 1792` sequence |
| GAN | PyTorch | frame generation | latent noise to synthetic `64 x 64` image |
| Autoencoder | PyTorch | reconstruction and anomaly analysis | `64 x 64` RGB frame |

## Action Classes

The project uses 25 action classes:

`ApplyEyeMakeup`, `ApplyLipstick`, `BabyCrawling`, `Basketball`, `BenchPress`, `Biking`, `BlowDryHair`, `Bowling`, `CliffDiving`, `CricketShot`, `SumoWrestling`, `Surfing`, `Swing`, `TableTennisShot`, `TaiChi`, `TennisSwing`, `ThrowDiscus`, `TrampolineJumping`, `Typing`, `UnevenBars`, `VolleyballSpiking`, `WalkingWithDog`, `WallPushups`, `WritingOnBoard`, `YoYo`

## Training Summary

### CNN and MLP

The CNN and MLP were trained from a single representative frame extracted from each video.

- frame chosen: middle frame of the video
- resize: `64 x 64`
- color format: RGB
- normalization: dataset mean and standard deviation computed from the training split

### RNN, LSTM, GRU, and BiLSTM + Attention

The recurrent models were trained on sequence features rather than raw frames.

- frames sampled every 5th frame
- frame resize: `128 x 128`
- normalization: ImageNet mean and standard deviation
- feature extractor 1: `ResNet34` truncated before classifier, output `512`
- feature extractor 2: `MobileNetV2` truncated before classifier, output `1280`
- concatenated feature vector per frame: `1792`
- fixed sequence length: `30`

So the final recurrent input is:

```text
(1, 30, 1792)
```

### GAN

The GAN was trained to generate realistic `64 x 64` action frames.

- generator input noise dimension: `128`
- output image size: `64 x 64`
- output range: `[-1, 1]`
- generator and discriminator trained with BCE loss

### Autoencoder

The autoencoder was trained to reconstruct `64 x 64` RGB frames.

- latent dimension: `128`
- output range: `[-1, 1]`
- reconstruction objective: MSE

## Deployment Notes

One important part of this project is that the deployment pipeline was aligned with the original training notebooks.

The final app now uses:

- the same middle-frame logic for CNN and MLP
- the same ResNet34 + MobileNetV2 feature pipeline for recurrent models
- the same GAN architecture as training
- the same autoencoder architecture as training

This alignment is critical because action-recognition models are highly sensitive to mismatches between training preprocessing and inference preprocessing.

## Streamlit Application Features

The deployed UI includes:

- model selection panel
- video upload and inference
- top-1 prediction and top-3 probabilities
- model-specific output cards
- GAN image generation preview
- autoencoder reconstruction comparison
- model diagnostics panel for selected models
- rich dashboard styling for demo and presentation

## Recommended Demo Flow

For presentation or evaluation, the best demo order is:

1. Start with `BiLSTM + Attention`
2. Show one or two correct action predictions
3. Compare with `CNN` or `MLP` as baseline models
4. Show `GAN` as a generative model
5. Show `Autoencoder` as a reconstruction model

The recommended primary action-recognition model for demonstration is:

```text
BiLSTM + Attention
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Required Model Files

Make sure these files are placed in the same folder as `app.py`:

| File | Purpose |
|---|---|
| `cnn_model.pth` | CNN weights |
| `mlp_model.pth` | MLP weights |
| `RNN.keras` | RNN sequence classifier |
| `LSTM.keras` | LSTM sequence classifier |
| `GRU.keras` | GRU sequence classifier |
| `BiLSTM+Attn.keras` | BiLSTM with attention classifier |
| `generator_final.pth` | GAN generator |
| `discriminator_final.pth` | GAN discriminator |
| `autoencoder_best.pth` | autoencoder weights |

## Requirements

The project uses:

- Python
- Streamlit
- NumPy
- OpenCV
- Pillow
- PyTorch
- Torchvision
- TensorFlow / Keras

Install everything from:

```bash
requirements.txt
```

## Key Technical Lessons

This project highlighted a few important deep learning deployment lessons:

- inference preprocessing must exactly match training preprocessing
- sequence models are usually stronger than single-frame models for action recognition
- attention improves temporal focus on informative frames
- GAN outputs are useful for visual understanding, but not for classification
- autoencoders are useful for reconstruction analysis and anomaly-style interpretation

## Limitations

- performance depends heavily on how similar the test video is to the training distribution
- social-media style videos may differ from dataset-style action videos
- GAN outputs are visually approximate and not meant for action prediction
- autoencoder reconstruction quality depends on training diversity

## Future Improvements

- add ensemble prediction across multiple classifiers
- save model metadata and fingerprints in a config file
- add batch video evaluation
- log confusion matrices in the app
- export prediction reports automatically

## 📥 Download Pretrained Models

Download models from:
https://drive.google.com/drive/folders/1oA5g2CPEBgjN02WTq5xJT_GwhMtCgIFi?usp=sharing

Place them in:
project/

## Academic Context

This project was created as part of a deep learning academic mini project on multi-model human action recognition and deployment.
