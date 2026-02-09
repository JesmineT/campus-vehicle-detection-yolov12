# Campus Vehicle Detection with YOLOv12s
This project implements a YOLOv12s-based deep learning model for detecting and classifying vehicles on a university campus. The system achieves high precision (75.97%) and strong generalization across 6 vehicle classes, trained on a custom dataset of 1,394 campus images.

**Achievements:**
- **mAP@50:** 71.29%
- **mAP@50:95:** 59.04%
- **Precision:** 75.97%
- **Recall:** 65.75%
- **Training Time:** ~17 minutes on A100 GPU

## Dataset
**Custom Campus Vehicle Dataset**
- **Total Images:** 1,394 images
- **Classes:** 6 vehicle categories
  - Honda City
  - Perodua Axia
  - Perodua Bezza
  - Proton Saga
  - Toyota Vios
  - Other Cars (catch-all category)

**Data Collection:**
- Captured under diverse real-world conditions
- Varied lighting: morning, afternoon, evening, night
- Varied angles and occlusions
- Compressed to 70% quality for efficient training

**Data Split:**
- Training: 70%
- Validation: 15%
- Test: 15%

**Annotation Process:**
- Hybrid auto-labeling + manual refinement using Roboflow
- YOLO format annotations
- Standardized resolution: 896×896 pixels

## Model Architecture
**YOLOv12s (Small variant)**
- Single-stage object detector
- Trained from scratch using `yolo12s.pt` configuration
- Input resolution: 896×896 pixels
- Built-in data augmentation via Ultralytics framework

**Why YOLOv12s?**
- Balanced speed and accuracy
- Excellent generalization on custom dataset
- High precision minimizes false positives
- Real-time inference capability

## Training Configuration
```python
Epochs: 200 (early stopped at 52)
Batch Size: 32
Image Size: 896×896
Workers: 8
Patience: 10
Learning Rate: 0.01 (initial and final)
Optimizer: AdamW
Momentum: 0.937
Weight Decay: 0.0005
Warmup Epochs: 5
```

## Performance Metrics
### Overall Performance
| Metric | Value |
|--------|-------|
| Precision | 0.7597 |
| Recall | 0.6575 |
| mAP@50 | 0.7129 |
| mAP@50:95 | 0.5904 |

### Per-Class Performance
| Class | Precision | Recall | mAP@50:95 |
|-------|-----------|--------|-----------|
| Perodua Axia | High | High | 0.699 |
| Perodua Bezza | High | High | 0.654 |
| Honda City | 0.831 | 0.547 | - |
| Other Cars | 0.619 | 0.751 | - |

**Insights:**
- Strongest performance on Perodua models (Axia, Bezza)
- High precision across most classes (~76% correct detections)
- Moderate recall indicates room for improvement in detection coverage
- Some confusion between specific models and "other_cars" category

## Data Augmentation
Built-in Ultralytics augmentations applied during training:
- **Mosaic Augmentation:** Combines 4 images into one
- **Random Scaling:** Simulates zoom variations
- **Random Translation:** Position variations
- **Horizontal Flipping:** Doubles effective dataset
- **Color Jittering:** HSV adjustments for lighting robustness

## Results Analysis
### Strengths
- **High Precision (75.97%):** Reliable detections, minimal false positives
- **Strong Performance on Specific Classes:** Perodua Axia and Bezza consistently well-detected
- **Good Generalization:** Solid performance on unseen test data
- **Fast Training:** Converged in 52 epochs (~17 minutes on A100)

### Limitations
- **Moderate Recall (65.75%):** Misses approximately 34% of vehicles
- **Class Confusion:** Some difficulty distinguishing "other_cars" from background
- **Honda City Performance:** Lower recall (54.7%) compared to other classes

### Confusion Matrix Insights
- **Best Classified:** Other_cars (294 true positives)
- **Most Confused:** Background ↔ Other_cars (62 false positives, 139 false negatives)
- **Bias:** Tendency to over-predict "other_cars" category

## Evaluation Curves
**Precision-Recall Curve:**
- Perodua Axia: AP = 0.782 (best)
- Perodua Bezza: AP = 0.781
- Honda City: AP = 0.629 (needs improvement)
- Overall mAP@0.5: 0.713

**F1-Confidence Curve:**
- Optimal threshold: 0.324 confidence
- Peak F1 score: 0.70

**Recall-Confidence Curve:**
- High recall (0.92) at low confidence
- Trade-off: Higher confidence reduces recall

## Training Optimizations
- **Early Stopping:** Prevented overfitting (stopped at epoch 52)
- **Learning Rate Scheduling:** Cosine decay from 0.01 to 0.01
- **Warmup Period:** First 5 epochs for stable initialization
- **Weight Decay:** L2 regularization (0.0005)
- **Multi-Worker Loading:** 8 workers for efficient data pipeline
