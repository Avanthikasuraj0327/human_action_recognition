# 🧠 GAN Architecture

## 📖 Overview
This model generates synthetic action frames using a Generative Adversarial Network.

## 🔷 Input
- Random noise vector

## 🔷 Generator
Noise → Dense → Upsampling / Transposed Conv → Image (64x64x3)

## 🔷 Discriminator
Image → Conv Layers → Sigmoid Output

## 🔷 Output
- Synthetic action-like images

## 🔷 Key Points
- Used for data generation
- Not used for classification
- Helps in data augmentation

# 🧠 Autoencoder Architecture

## 📖 Overview
This model reconstructs input frames and measures reconstruction error.

## 🔷 Input
- Image: 64x64x3

## 🔷 Encoder
Conv → Conv → Latent Representation

## 🔷 Decoder
Latent → Upsampling / Transposed Conv → Reconstructed Image

## 🔷 Output
- Reconstructed frame
- Reconstruction error (MSE)

## 🔷 Key Points
- Used for anomaly detection
- Learns compressed representation
- Not used for direct classification
