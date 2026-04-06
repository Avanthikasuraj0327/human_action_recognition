📌 Project Title

Human Action Recognition using GAN-based Autoencoder

📖 Overview

This project implements a GAN-based Autoencoder (GAN-AE) for human action recognition. The model learns to reconstruct input video frames and uses reconstruction error to distinguish between different actions.

⚙️ Methodology
Extract frames from videos
Preprocess (resize, normalize)
Train:
Generator (Autoencoder)
Discriminator (real vs reconstructed frames)
Use reconstruction loss for action understanding
🧠 Model Architecture
Encoder → compress spatial features
Decoder → reconstruct frames
Discriminator → adversarial training
📂 Input
Video frames from UCF101 (subset of 25 classes)
📤 Output
Reconstructed frames
Loss curves (Generator & Discriminator)
Action classification (based on reconstruction patterns)
📊 Evaluation Metrics
Reconstruction Loss
PSNR (optional)
SSIM (optional)
🚀 How to Run
pip install -r requirements.txt

Run notebook:

jupyter notebook gan-ae-action-recognition.ipynb
📌 Key Features
Unsupervised / semi-supervised learning
Works well with limited labeled data
