# 🧠 CNN Architecture

## 📖 Overview
This model performs human action recognition using a single representative frame from the video.

## 🔷 Input
- Middle frame of video
- Image size: 64x64x3 (RGB)

## 🔷 Preprocessing
- Resize to 64x64
- Normalize pixel values

## 🔷 Architecture
Conv2D → ReLU → MaxPooling  
Conv2D → ReLU → MaxPooling  
Flatten  
Fully Connected Layer  
Softmax Output Layer  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- Captures spatial features only
- Does not model temporal information
- Serves as a strong baseline


# 🧠 MLP Architecture

## 📖 Overview
This model uses a flattened image input and classifies actions using fully connected layers.

## 🔷 Input
- Middle frame of video
- Image size: 64x64x3

## 🔷 Preprocessing
- Resize to 64x64
- Flatten into 1D vector

## 🔷 Architecture
Flatten  
Dense → ReLU  
Dropout  
Dense → ReLU  
Dense → Softmax  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- No spatial feature extraction
- Simple and computationally efficient
- Lowest performance among models
