# CAT-CNN: Complex-Aware Transformer-CNN for Refractive Index Prediction

## Overview
CAT-CNN is a deep learning model that predicts the real and imaginary parts of the refractive index based on the width and height of a rectangular waveguide. The model integrates Convolutional Neural Networks (CNNs) for feature extraction, Squeeze-and-Excitation (SE) blocks for channel-wise attention, and Transformer blocks for capturing global dependencies. It is designed to enhance predictive accuracy over conventional models such as CNN-LSTM.

## Features
- **Hybrid CNN-Transformer Architecture**: Combines local feature extraction with global sequence modeling.
- **Multi-Task Learning**: Predicts both real and imaginary components of the refractive index.
- **Normalization and Scaling**: Standardizes input features for better model performance.
- **Cosine Decay Learning Rate**: Adaptive learning rate for efficient training.
- **Custom Loss Weighting**: Prioritizes real and imaginary parts differently for balanced optimization.

## Dataset
The model is trained on a dataset containing:
- **Input Features**: Width (`w_Si(nm)`) and Height (`h_Si(nm)`) of the rectangular waveguide.
- **Output Targets**: Real and imaginary parts of the refractive index.
- **Preprocessing**: Standardization using `StandardScaler` for improved convergence.

## Model Architecture
1. **CNN Feature Extraction**:
   - Uses multiple `Conv1D` layers with `Swish` activation for feature learning.
   - Batch normalization and dropout for regularization.
   - Squeeze-and-Excitation (SE) block enhances feature importance.

2. **Transformer Blocks**:
   - Multi-Head Attention (MHA) to capture long-range dependencies.
   - Layer normalization and residual connections for stability.
   - Feed-forward layers with Swish activation.

3. **Prediction Layers**:
   - Fully connected layers process the extracted features.
   - Separate output heads for real and imaginary refractive index components.

## Installation & Dependencies
```bash
pip install tensorflow numpy pandas scikit-learn
```

## Training & Evaluation
- **Training**:
  - 500 epochs with batch size 32.
  - Adam optimizer with cosine decay learning rate scheduling.
  - Multi-task loss with different weightings (MSE for real, Huber for imaginary part).

- **Evaluation Metrics**:
  - Mean Absolute Error (MAE) for both outputs.
  - R² Score for real and imaginary parts.
  - Weighted combined R² metric for overall performance assessment.



## Future Improvements
- **Hyperparameter Optimization**: Fine-tuning kernel sizes, number of transformer heads, and SE block ratios.
- **Data Augmentation**: Generating synthetic waveguide configurations to enhance generalization.
- **Alternative Architectures**: Exploring Swin Transformer or Graph Neural Networks (GNNs) for better spatial understanding.
- **Physics-Informed Training**: Incorporating domain-specific constraints to refine predictions.



