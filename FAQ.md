# Neural Network Training and Implementation FAQ

This document provides answers to common questions and issues experienced when training and implementing neural network pipelines.

## Table of Contents
- [Training Issues](#training-issues)
  - [Overfitting](#overfitting)
  - [Underfitting](#underfitting)
  - [Vanishing Gradients](#vanishing-gradients)
  - [Exploding Gradients](#exploding-gradients)
  - [Slow Training](#slow-training)
  - [Loss Not Decreasing](#loss-not-decreasing)
- [Implementation Issues](#implementation-issues)
  - [Data Loading](#data-loading)
  - [Model Architecture](#model-architecture)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Memory Issues](#memory-issues)
  - [Reproducibility](#reproducibility)
- [Debugging Tips](#debugging-tips)
- [Best Practices](#best-practices)

---

## Training Issues

### Overfitting

**Q: My model performs well on training data but poorly on validation/test data. What's wrong?**

A: This is classic overfitting. Your model is memorizing the training data instead of learning generalizable patterns.

**Solutions:**
- Add regularization (L1, L2, or both)
- Use dropout layers
- Reduce model complexity (fewer parameters)
- Get more training data
- Use data augmentation
- Implement early stopping
- Try ensemble methods

**Example:**
```python
# Add dropout in PyTorch
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(50, 10)
)
```

### Underfitting

**Q: My model performs poorly on both training and validation data. What should I do?**

A: Your model is underfitting - it's too simple to capture the patterns in your data.

**Solutions:**
- Increase model complexity (more layers or neurons)
- Train for more epochs
- Reduce regularization
- Use more complex feature engineering
- Check for data quality issues
- Ensure learning rate isn't too low

### Vanishing Gradients

**Q: My deep network trains very slowly or stops improving after a few epochs. What's happening?**

A: You're likely experiencing vanishing gradients, where gradients become extremely small in earlier layers during backpropagation.

**Solutions:**
- Use activation functions like ReLU, LeakyReLU, or ELU instead of sigmoid or tanh
- Implement batch normalization
- Use residual connections (ResNet architecture)
- Apply proper weight initialization (Xavier/He initialization)
- Consider using LSTM/GRU for RNNs instead of vanilla RNNs

**Example:**
```python
# Use He initialization in PyTorch
import torch.nn.init as init

for m in model.modules():
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
```

### Exploding Gradients

**Q: My loss becomes NaN or extremely large during training. What's the issue?**

A: You're experiencing exploding gradients, where gradients become too large during backpropagation.

**Solutions:**
- Implement gradient clipping
- Reduce learning rate
- Use batch normalization
- Check weight initialization
- Verify data preprocessing (normalize/standardize inputs)
- Use a more stable optimizer (Adam instead of SGD)

**Example:**
```python
# Gradient clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Slow Training

**Q: My model takes too long to train. How can I speed it up?**

A: Several factors can slow down training.

**Solutions:**
- Increase batch size (if memory allows)
- Use GPU acceleration
- Implement mixed precision training
- Use more efficient data loading (DataLoader with multiple workers)
- Profile your code to find bottlenecks
- Consider using a simpler model architecture
- Use learning rate warmup and scheduling

**Example:**
```python
# PyTorch DataLoader with multiple workers
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

### Loss Not Decreasing

**Q: My loss isn't decreasing at all. What could be wrong?**

A: Several issues could cause this.

**Solutions:**
- Check learning rate (try different values: 1e-2, 1e-3, 1e-4)
- Verify data preprocessing and normalization
- Check for bugs in loss function
- Ensure labels are correctly formatted
- Try different optimizers
- Check for dead ReLU neurons (try LeakyReLU)
- Verify gradient flow (use gradient checking)
- Simplify the model first to ensure basic training works

---

## Implementation Issues

### Data Loading

**Q: What are common mistakes in data loading and preprocessing?**

A: Data issues are the most common source of bugs in neural networks.

**Common Issues and Solutions:**
- **Not normalizing data**: Always normalize/standardize inputs
  ```python
  # Standardize features
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)  # Use same scaler!
  ```

- **Data leakage**: Fit preprocessing only on training data
- **Wrong data shapes**: Check dimensions match model expectations
- **Shuffling**: Shuffle training data but not validation/test data
- **Class imbalance**: Use weighted loss or sampling strategies

### Model Architecture

**Q: How do I choose the right model architecture?**

A: Start simple and gradually increase complexity.

**Guidelines:**
- Start with a simple baseline model
- Use proven architectures for your domain (ResNet for vision, LSTM/Transformer for text)
- Match architecture to problem:
  - Classification: Softmax output layer
  - Regression: Linear output layer
  - Binary classification: Sigmoid output with 1 neuron or Softmax with 2 neurons
- Consider model size vs. available data (large models need more data)

**Common Architecture Patterns:**
```python
# Simple classification network
model = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, num_classes)
)
```

### Hyperparameter Tuning

**Q: How should I tune hyperparameters?**

A: Systematic tuning is better than random changes.

**Recommended Approach:**
1. **Learning rate** - Most important! Try: 1e-2, 1e-3, 1e-4, 1e-5
2. **Batch size** - Try: 16, 32, 64, 128
3. **Model architecture** - Number of layers and neurons
4. **Regularization** - Dropout rates, L2 penalty
5. **Optimizer** - Adam is a good default, try SGD with momentum too

**Tools:**
- Grid search for small parameter spaces
- Random search for larger spaces
- Bayesian optimization (e.g., Optuna, Hyperopt)
- Learning rate finder

**Example:**
```python
# Learning rate finder pattern
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for lr in lrs:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train for a few epochs and track loss
```

### Memory Issues

**Q: I'm getting CUDA out of memory errors. How can I fix this?**

A: GPU memory is limited and needs careful management.

**Solutions:**
- Reduce batch size
- Use gradient accumulation to simulate larger batches
- Use mixed precision training (fp16)
- Clear cache regularly: `torch.cuda.empty_cache()`
- Use gradient checkpointing for very deep networks
- Reduce model size
- Use smaller input sizes (resize images)

**Example:**
```python
# Gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Reproducibility

**Q: My results vary between runs. How can I ensure reproducibility?**

A: Random seeds and environment factors affect reproducibility.

**Solution:**
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Note:** Some operations are non-deterministic by design for performance reasons.

---

## Debugging Tips

### General Debugging Strategy

1. **Start simple**: Test with a tiny dataset first (10-100 samples)
2. **Overfit on purpose**: Ensure your model can memorize a small batch
3. **Check shapes**: Print tensor shapes at each layer
4. **Visualize**: Plot loss curves, activations, gradients
5. **Unit test components**: Test data loading, model forward pass separately
6. **Use assertions**: Add shape checks in your code
7. **Compare with baselines**: Implement a simple baseline first

### Common Debugging Commands

```python
# Check model summary
print(model)

# Check tensor shapes
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

# Check for NaN values
assert not torch.isnan(loss).any(), "Loss is NaN!"

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

---

## Best Practices

### Training Workflow

1. **Data preparation**
   - Split into train/validation/test sets
   - Normalize/standardize features
   - Handle missing values
   - Address class imbalance if present

2. **Model development**
   - Start with simple baseline
   - Gradually increase complexity
   - Use proven architectures when possible
   - Implement proper evaluation metrics

3. **Training**
   - Monitor both training and validation metrics
   - Use early stopping
   - Save checkpoints regularly
   - Log hyperparameters and results

4. **Evaluation**
   - Test on held-out test set only once
   - Use multiple metrics (accuracy, F1, precision, recall)
   - Analyze errors and failure cases
   - Consider cross-validation for small datasets

### Code Organization

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── __init__.py
│   └── model.py
├── training/
│   ├── __init__.py
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
├── configs/
│   └── config.yaml
└── main.py
```

### Documentation and Tracking

- Document all experiments with hyperparameters and results
- Use experiment tracking tools (TensorBoard, Weights & Biases, MLflow)
- Version control your code
- Save model configurations and weights
- Create reproducible environments (requirements.txt, Docker)

### Performance Optimization

1. **Data loading**: Use efficient data loaders with prefetching
2. **Computation**: Use GPU when available, batch operations
3. **Memory**: Monitor GPU memory, use gradient accumulation if needed
4. **Mixed precision**: Use automatic mixed precision (AMP) for faster training

### Common Pitfalls to Avoid

- ❌ Not shuffling training data
- ❌ Fitting preprocessing on test data
- ❌ Tuning hyperparameters on test set
- ❌ Not normalizing inputs
- ❌ Using same random seed for all splits
- ❌ Not checking for data leakage
- ❌ Ignoring class imbalance
- ❌ Not setting random seeds for reproducibility
- ❌ Training too long without early stopping
- ❌ Not monitoring validation metrics

---

## Additional Resources

### Learning Materials
- [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Fast.ai Course](https://course.fast.ai/)

### Tools and Libraries
- **PyTorch**: Popular deep learning framework
- **TensorFlow/Keras**: Alternative framework
- **Scikit-learn**: Preprocessing and traditional ML
- **TensorBoard**: Visualization tool
- **Weights & Biases**: Experiment tracking
- **Optuna**: Hyperparameter optimization

### Communities
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/neural-network)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Cross Validated](https://stats.stackexchange.com/)

---

**Contributing**: If you have additional questions or solutions to add, please submit a pull request or open an issue.

**Last Updated**: October 2025
