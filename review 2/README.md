# Human Action Recognition using RNN, LSTM, and GRU

## 📖 Overview
This project focuses on temporal modeling of human actions using sequence models such as RNN, LSTM, and GRU.

## ⚙️ Methodology
- Extract sequences of frames from videos
- Convert frames into feature vectors
- Feed sequences into RNN/LSTM/GRU models
- Perform classification

## 🧠 Model Architecture
- Feature extraction layer (CNN or preprocessing)
- Sequence models:
  - RNN
  - LSTM
  - GRU
- Fully connected output layer

## 📂 Input
- Sequence of frames per video

## 📤 Output
- Predicted action class
- Model comparison results

## 📊 Evaluation Metrics
- Accuracy
- Loss curves
- Model comparison

## 🚀 How to Run
pip install -r requirements.txt  
jupyter notebook action-recognition-lstm-rnn-gru.ipynb

## 📌 Key Features
- Captures temporal dependencies
- LSTM and GRU handle long-term motion patterns
- Better performance for video-based tasks
