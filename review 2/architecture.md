# 🧠 RNN Architecture

## 📖 Overview
This model captures temporal dependencies in video sequences using a simple RNN.

## 🔷 Input
- Sequence of 30 frames
- Feature vector per frame: 1792

## 🔷 Feature Extraction
- ResNet34 → 512 features
- MobileNetV2 → 1280 features
- Concatenated → 1792

## 🔷 Architecture
Sequence Input (30, 1792)  
Simple RNN  
Dense Layer  
Softmax Output  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- Captures temporal information
- Suffers from vanishing gradient problem
- Lower performance than LSTM/GRU

# 🧠 LSTM Architecture

## 📖 Overview
This model uses Long Short-Term Memory (LSTM) to capture long-term temporal dependencies.

## 🔷 Input
- Sequence: (30, 1792)

## 🔷 Feature Extraction
- ResNet34 + MobileNetV2 combined features

## 🔷 Architecture
Sequence Input  
LSTM Layer  
Dense Layer  
Softmax Output  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- Handles long-term dependencies
- Performs better than RNN
- Suitable for video-based tasks

# 🧠 GRU Architecture

## 📖 Overview
This model uses Gated Recurrent Units (GRU) for efficient temporal modeling.

## 🔷 Input
- Sequence: (30, 1792)

## 🔷 Feature Extraction
- ResNet34 + MobileNetV2 features

## 🔷 Architecture
Sequence Input  
GRU Layer  
Dense Layer  
Softmax Output  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- Faster than LSTM
- Similar performance to LSTM
- Fewer parameters

# 🧠 BiLSTM + Attention Architecture

## 📖 Overview
This model enhances LSTM by using bidirectional learning and attention mechanism.

## 🔷 Input
- Sequence: (30, 1792)

## 🔷 Feature Extraction
- ResNet34 + MobileNetV2 combined features

## 🔷 Architecture
Sequence Input  
Bidirectional LSTM  
Attention Layer  
Context Vector  
Dense Layer  
Softmax Output  

## 🔷 Output
- 25 action classes

## 🔷 Key Points
- Learns from both past and future context
- Attention focuses on important frames
- Best performing model
