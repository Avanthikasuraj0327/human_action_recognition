# Human Action Recognition using GAN-Based Autoencoder

## 📖 Overview
This project implements a GAN-based Autoencoder (GAN-AE) for human action recognition. The model learns to reconstruct video frames and uses reconstruction patterns to understand different actions.

## ⚙️ Methodology
- Extract frames from videos
- Preprocess images (resize, normalize)
- Train Generator (Autoencoder) and Discriminator
- Use reconstruction loss for learning action features

## 🧠 Model Architecture
- Encoder: Extracts latent features
- Decoder: Reconstructs frames
- Discriminator: Distinguishes real vs generated frames

## 📂 Input
- Video frames from UCF101 dataset (25 classes subset)

## 📤 Output
- Reconstructed frames
- Generator and Discriminator loss
- Learned action representations

## 📊 Evaluation Metrics
- Reconstruction Loss
- PSNR (optional)
- SSIM (optional)

## 🚀 How to Run
pip install -r requirements.txt  
jupyter notebook gan-ae-action-recognition.ipynb

## 📌 Key Features
- Works with limited labeled data
- Captures spatial patterns
- Uses adversarial learning
