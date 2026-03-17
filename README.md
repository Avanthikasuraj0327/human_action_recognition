# Human Action Recognition using CNN + RNN/LSTM/GRU

## Overview

This project focuses on recognizing human actions in videos using a hybrid deep learning architecture combining **CNN (ResNet34)** for spatial feature extraction and **RNN/LSTM/GRU** for temporal sequence modeling.

---

## Methodology

1. **Frame Extraction**

   * Videos are split into frames using OpenCV

2. **Feature Extraction**

   * ResNet34 (pretrained on ImageNet) extracts spatial features

3. **Sequence Modeling**

   * RNN / LSTM / GRU captures temporal dependencies

4. **Classification**

   * Fully connected layer predicts action class

---

## Architecture

Video → Frames → ResNet34 → Feature Vectors → LSTM/GRU → FC Layer → Prediction

---

## Results

| Model | Accuracy |
| ----- | -------- |
| RNN   | XX%      |
| LSTM  | XX%      |
| GRU   | XX%      |

---

## Project Structure

```
src/
    dataset.py
    model.py
    train.py
    evaluate.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

##  Run Training

```bash
python src/train.py
```

---

##  Evaluation

```bash
python src/evaluate.py
```

---

##  Future Improvements

* Attention mechanism
* Transformer-based models
* Real-time deployment using FastAPI
* Integration with CCTV surveillance systems

---

##  Author

Avanthika K S
M.Tech AI, Amrita Vishwa Vidyapeetham
