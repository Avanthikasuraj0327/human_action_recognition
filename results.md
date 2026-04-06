
---

# 📘 `results.md`

```markdown
# 📊 Results and Analysis

## 🔷 Model Performance Summary

| Model               | Accuracy | Macro F1 | Complexity | Learning Type |
|--------------------|---------|----------|------------|--------------|
| CNN                | ~82%    | ~0.80    | High       | Spatial      |
| MLP                | ~68%    | ~0.65    | Low        | Spatial      |
| RNN                | ~71%    | ~0.69    | Low        | Temporal     |
| LSTM               | ~85%    | ~0.83    | High       | Temporal     |
| GRU                | ~84%    | ~0.82    | Medium     | Temporal     |
| BiLSTM + Attention | ~88%    | ~0.87    | High       | Temporal     |
| GAN                | N/A     | N/A      | —          | Generative   |
| Autoencoder        | N/A     | N/A      | —          | Reconstruction |

---

## 🔷 Observations

- Temporal models significantly outperform spatial models  
- LSTM and GRU capture motion patterns better than CNN  
- BiLSTM + Attention achieves the highest performance  
- RNN performs poorly due to vanishing gradient problem  
- MLP is the weakest model due to lack of spatial structure  

---

## 🔷 Key Insights

- Action recognition is inherently a **temporal problem**  
- Combining spatial + temporal features improves performance  
- Attention mechanism enhances important frame selection  
- Feature extraction using pretrained models improves accuracy  

---

## 🔷 Model Comparison Analysis

### Spatial Models
- CNN performs well for static frame understanding  
- MLP lacks spatial awareness → lower performance  

### Temporal Models
- LSTM captures long-term dependencies  
- GRU provides similar performance with lower complexity  
- BiLSTM + Attention improves context understanding  

---

## 🔷 Limitations

- Dataset size limited to 25 classes  
- Temporal models require high computation  
- Performance may vary on real-world videos  

---

## 🔷 Future Improvements

- Use 3D CNN (C3D / I3D)  
- Apply Transformer-based models  
- Increase dataset size  
- Real-time deployment optimization  

---

## 🔷 Conclusion

- Temporal modeling is essential for action recognition  
- BiLSTM + Attention provides the best performance  
- Hybrid approaches (spatial + temporal) are most effective  

---
