# Human Action Recognition using CNN + MLP

## 📖 Overview
This project uses a Convolutional Neural Network (CNN) for feature extraction and a Multi-Layer Perceptron (MLP) for classification of human actions.

## ⚙️ Methodology
- Extract frames from videos
- Pass frames through CNN
- Flatten features
- Classify using MLP

## 🧠 Model Architecture
- Convolutional layers (feature extraction)
- Fully connected layers (classification)
- Softmax output

## 📂 Input
- Image frames from UCF101 dataset

## 📤 Output
- Predicted action labels
- Accuracy and loss graphs

## 📊 Evaluation Metrics
- Accuracy
- Cross-entropy loss

## 🚀 How to Run
pip install -r requirements.txt  
jupyter notebook cnn-mlp-action-recognition-dl.ipynb

## 📌 Key Features
- Simple and fast baseline model
- Focuses on spatial information
- Easy to train and interpret
