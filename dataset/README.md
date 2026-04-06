# UCF101 Dataset (Subset - 25 Classes)

## 📖 Overview
This project uses a subset of the UCF101 dataset containing 25 human action classes for efficient training and experimentation.

## 📂 Dataset Details
- Dataset: UCF101
- Classes Used: 25
- Data Type: Video clips
- Format: AVI / extracted frames

## 🎯 Example Classes
- ApplyEyeMakeup  
- ApplyLipstick  
- Archery  
- BabyCrawling  
- BalanceBeam  
- BaseballPitch  
- Basketball  
- BenchPress  
- Biking  

## ⚙️ Preprocessing
- Frame extraction using OpenCV
- Resize to fixed dimensions (e.g., 224x224)
- Normalize pixel values
- Create sequences for temporal models

## 📊 Dataset Split
- Training: 70%
- Validation: 15%
- Testing: 15%

## 📁 Folder Structure
dataset/
│── class_1/
│   ├── video1/
│   ├── video2/
│── class_2/
│── ...

## 🚀 How to Prepare Data
python preprocess.py

## 📌 Why Subset?
- Reduces computational cost
- Faster training
- Suitable for experimentation and prototyping
