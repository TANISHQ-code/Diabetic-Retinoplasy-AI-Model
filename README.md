# Diabetic Retinopathy AI Model

An AI-based medical image classification system for automated diabetic retinopathy severity detection from retinal fundus images.

## Features
- EfficientNet-B3 Transfer Learning
- Image preprocessing using CLAHE + Gaussian Blur
- Data augmentation pipeline
- Focal Loss for imbalanced classification
- Mixed Precision GPU Training
- Two-stage Fine Tuning
- CNN + SVM / MLP benchmarking
- Quantum-Classical Hybrid QCNN + QSVM experiments

## Tech Stack
Python, TensorFlow, PyTorch, OpenCV, Scikit-learn, PennyLane, NumPy, Pandas

## Evaluation Metrics
- Accuracy
- F1 Score
- ROC-AUC
- Confusion Matrix
- Cohen Kappa Score
## Model Performance

| Model | Accuracy | F1 Score |
|---------|----------|-----------|
| CNN + SVM | 46.53% | 44.73% |
| CNN + MLP | 44.00% | 42.60% |
| QCNN + SVM | 22.10% | 20.30% |

Best performing model: CNN + SVM
## Experimental Results

### Best Performing Model (CNN + SVM)

| Metric | Score |
|----------|----------|
| Test Accuracy | 46.53% |
| Test F1 Score | 44.73% |
| Validation Accuracy | 53.73% |
| Validation F1 Score | 53.22% |

![Best SVM Results](results/svm_best_results.png.jpeg)

---

### CNN + MLP Benchmark

| Metric | Score |
|----------|----------|
| Accuracy | 44.0% |
| Macro F1 Score | 42.6% |

![CNN MLP Results](results/cnn_mlp_results.png.jpeg)