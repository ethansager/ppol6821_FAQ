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
  - [NaN issues](#nan-issues)
- [Metrics](#metrics)
- [Debugging Tips](#debugging-tips)
- [Best Practices](#best-practices)

---

## Training Issues

### Overfitting

**Q: My model performs well on training data but poorly on validation/test data. What's wrong?**

A: Try revisiting your pre-processing steps. In pytorch, this would be the torchvision. transforms. Here, you can try a wide range of combinations of random flips, rotations, inverts, crops, blackouts and color adjustments. The most important part, however, is the actual normalization of mean and standard deviation. If you are working on images, don't forget to run the tensor transformation in this step as well.

Also, overly complex models could negatively affect your fitness instead. For instance, try to use fewer units when designing an LSTM network if you find significant overfitting. Or, if using CNN on pictures, too many convolutional layers/filters/dense layers would also lead to overfitting instead of precision. So when overfitting occurs, playing around with the complexity of the model is sometimes a good way to start.

**Q: What are some strategies to prevent overfitting?**

A: Strategies to prevent overfitting can be data-delated, model-related, or training-related.

**Preventing overfitting through your data:**
Try increasing the size of your training data. Having large data with a good amount of variation improves the model’s ability to learn patterns. One possibility for increasing the size of the training data is data augmentation. You can create artificial data that increases the size of your dataset and adds more variation. 

**Preventing overfitting through your model:**
First, add or increase dropout. Dropout randomly turns off a percentage of neurons during training, forcing the network to learn the most important features. Second, reduce model complexity by removing layers or the number of neurons in each layer. Complex models are more likely to experience overfitting. Third, use regularization. L1 and L2 regularization penalizes the loss function more for larger weights, helping to simplify the model.

**Preventing overfitting through training process:**
First, implement early stopping. This will stop training when the validation loss starts to increase. Second, utilize cross-validation, which mimics the validation step by testing the model on unseen subsets of the data.

### Underfitting

**Q: My model performs poorly on both training and validation data. What should I do?**

A: Try to increase model your architecure, the current one may not be complex enough. However, it can also be a sign of gradient collaps. In the context of CNNs, for example, introducing a weight initalization as part of the architecture enures the weight are loaded correctly. During gradient/model collaps, a model that should show moderately high accuracy, can indicate single digit accuracy.;

### Vanishing Gradients

**Q: My deep network trains very slowly or stops improving after a few epochs. What's happening?**

A: You’re experiencing vanishing gradients. This is a problem where the gradients become extremely small as they undergo backpropagation during training. This causes slow tuning of the weights in the early layers, preventing learning and halting the training process.

**Q: How can I solve the problem of vanishing gradients?**

A: If you are using sigmoid or tanh activation functions, consider alternatives. The derivatives of these functions are small, which can lead to vanishing gradients when used across many layers. You can try to introduce a different activation function, such as ReLU. Additionally, implementing Batch Normalization can be helpful.

### Exploding Gradients

**Q: My loss becomes NaN or extremely large during training. What's the issue?**

A: You're experiencing exploding gradients, where gradients become too large during backpropagation. This happens when the product of many derivatives grows exponentially during backpropagation. The model fails to converge and parameters turn into NaN (not a number) because they are too large.

**Q: How can I solve the problem of exploding gradients?**

A: Gradient clipping is the most common solution. It sets a maximum threshold for the gradients. L1 or L2 regularization penalizes large weights, which can also help the problem of exploding gradients. You can also consider changing your architecture. LSTM networks have mechanisms that help resolve the problem when using sequential data.


### Slow Training

**Q: My model takes too long to train. How can I speed it up?**

A: Even if everything is done correctly, this can be a standard issue. Your model may just be way too deep and complex for the local machine you are using. If possible, you can move the script into Google Colab and use the GPUs, which speed up training time compared to a local CPU. If still slow, try creating local Drive copies of your paths. The setup will take some time and you will have to set it up each time you connect to a GPU, but it will further speed up training time.

You can go about making the copies in two ways either make either in bash or in python
```bash
# Copy folder from Drive
!mkdir -p Dogs
!cp -r "/content/drive/MyDrive/Dogs" Dogs
```

```python
import shutil
import os
# Copy folder from Drive 
src = "/content/drive/MyDrive/Dogs"
dst = "/content/Dogs"
os.mkdir(dst)
shutil.copytree(src, dst, dirs_exist_ok=True)
```

### Loss Not Decreasing

**Q: My loss isn't decreasing at all. What could be wrong?**

A: If the loss is vibrating heavily but not decreasing, check the learning rate first. Turbulent, non-converging loss is probably due to a high learning rate. You may also consider using AdamW instead of Adam or SGD as the optimizer for more stable loss convergence. You may also check your model design to see whether you need a deeper network with larger batch sizes, more filters, layers, and residual connectors.

## Implementation Issues

### Data Loading

**Q: How can I get access to a massive external database on HuggingFace?**
**A:** Actually accessing a dataset from an external database is unfortunately not as simple as just loading it in or directly streaming it without first accessing the platform; without going through the proper channels, your connection will not be allowed by the platform or may be extremely memory intensive. There are a few considerations to make here:
1) Prior to calling the dataset, you must first create a login on HuggingFace’s platform, and then create an ‘access token’. This will give you an access key with a name and passcode that are visible to you a single time. If using Google Colab, you can input them in the ‘secrets tab’ (denoted with a key symbol) in the workbook you’re working out of. From here, you can input the following code to complete this connection:
```python
from google.colab import userdata
from huggingface_hub import login


# Retrieve the Hugging Face token from Colab secrets
hf_token = userdata.get('HF_TOKEN')


# Login to Hugging Face Hub programmatically
if hf_token:
    login(token=hf_token)
    print("Hugging Face token configured successfully.")
else:
    print("HF_TOKEN not found in Colab secrets. Please add it.")
```
2) Loading the entirety of a massive dataset can cause memory errors and large loading times; as such, streaming the data is a more memory-efficient solution, as it fetches data on demand rather than loading everything in regardless of available memory.
```python
try:
            print("\n🔍 Attempting to load with streaming and smart filtering...") # Progress checker - notes that you’ve reached this stage
            dataset_stream = load_dataset(
                "[dataset title]",
                streaming=True,
                split='train'
            )
```
**Q: What are common mistakes in data loading and preprocessing?**

**A:** Not running a normalization step is the biggest issue in pre-processing. If you work with images, don't forget the Tensor transformation as well. For data-loading, this depends on your context. If you designed your own DataLoader function, try adjusting the batch size and make sure you set `shuffle=True` for the training set.

**Additional common mistakes:**
- **Memory leaks**: Check that you're not accumulating gradients unintentionally (use `with torch.no_grad()` during validation/testing)
- **Data leakage**: Ensure your normalization statistics (mean/std) are computed only on the training set, then applied to validation/test sets
- **Inconsistent preprocessing**: Apply the exact same transformations during training and inference (except data augmentation)
- **Wrong data types**: Verify that your inputs are `float32` and labels match your loss function requirements (e.g., `long` for CrossEntropyLoss)
- **num_workers issues**: If using multiple workers in DataLoader, start with `num_workers=0` to debug, then increase for performance
- **Incorrect batch dimensions**: Check that your input shape matches what the model expects (e.g., `[batch_size, channels, height, width]` for CNNs)

**Q: My training is very slow. Could this be a data loading issue?**

**A:** First, make sure you've enabled GPU in Colab (`Runtime > Change runtime type > GPU`). If you're already using GPU but training is still slow, then data loading could be the issue. Check if your GPU is underutilized:
```python
# In Colab, check GPU usage
!nvidia-smi
```

If GPU utilization is low (<50%), try these Colab-specific solutions:
- **Increase batch size**: Colab GPUs have good memory; try `batch_size=64` or `128`
- **Use `pin_memory=True`**: Speeds up CPU-to-GPU transfer
- **Keep `num_workers=2`**: Colab has limited CPU cores, so higher values (4-8) may actually slow things down or cause crashes. Start with 2.
- **Load data**: If using Google Drive, copy datasets to Colab's local storage first:
```python
# Copy from Drive to local (much faster)
!cp -r /content/drive/MyDrive/dataset /content/dataset

# Then load from local path
train_dataset = YourDataset('/content/dataset')
```

- **Avoid slow transformations**: Heavy augmentations on CPU can bottleneck, so consider simpler transforms or pre-augment your data

**Optimal Colab DataLoader setup:**
```python
train_loader = DataLoader(dataset, 
                         batch_size=64,      # Larger for Colab GPUs
                         shuffle=True,
                         num_workers=2,       # Lower for Colab
                         pin_memory=True)
```

**Note**: If training is still slow after GPU is enabled, it might be your model architecture, not data loading.

**Q: Should I shuffle my validation and test sets?**

**A:** No, always set `shuffle=False` for validation and test sets. Shuffling can:
- Make debugging harder (non-reproducible results)
- Complicate tracking of specific samples
- Waste computation (unnecessary randomization)

The only dataset that should have `shuffle=True` is your training set, to prevent the model from learning order-dependent patterns.

### Model Architecture

**Q: How do I choose the right model architecture?**

A: There is no right architecture, this depends on your context, use case and objectives. As a rule of thumb, design a minimum model first, then an all-out maximum model and inspect results and training times. Then, converge on something in between.

### Hyperparameter Tuning

**Q: How should I tune hyperparameters?**

### Reproducibility

**Q: My results vary between runs. How can I ensure reproducibility?**

A: The first thing you would want to check is that you are properly implementing seeds. Computers are not really random and must be given a starting point in which they build from this is what a seed does. Random seeds and environment factors affect reproducibility, it is reccomended not to use something that is particular to you say your birthday etc https://www.random.org/ can help with coming up with seeds. You can see some code how to implement this below. The second issue you may have espeically with notebooks is the order of code execution always check before training that you can run the code up till this point in a single call that makes sure you are not operating on a object that can not be reproduced. 

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

# or in keras
import keras
keras.utils.set_random_seed(42)

```
see: https://keras.io/examples/keras_recipes/reproducibility_recipes/ if you want more technical explantion. 
### NaN Issues

**I am getting NaN errors during the training phase. What's going on?**

A: NaN means you have missing data. For example, if you are training a CNN model and have a single observation with a missing value for one variable, your model will result in an NaN error. You can use a variety of tools to identify missingness across your variables to find the culprit. For an LSTM model, you have options such as imputation to fill in missing values, but your decisions need to be grounded in theory, best practice, etc.

---

## Metrics

**Q. In my classification model, I feel like loss and accuracy aren't giving me the full picture. What else can I look at to judge how my model would fare depending on my intended use case?**

### Accuracy
The most common metric evaluating ML and NN classification models, accuracy simply tells us how often the model is 'right' out of all its predictions. Formally,

$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Number of total predictions}}$ 

For two-class classification:

$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP +FN}$

For multi-class classification, each class has its own accuracy relative to the rest. It can be worth looking at these individually, since a hight overall accuracy score can obscure poor performance on minority classes.

Accuracy is popular because it's intuitive, but it doesn't always give the best sense of how a model will perform in its intended practical application. For instance a system meant to give a medical diagnosis of a serious condition, a false negative (someone who 'slips through the net' and doesn't receive the treatment they need) is a much bigger problem than a false positive (a healthy person flagged by the model). In a scenario where most of the subjects tested are healthy, false negatives can easily fly under the radar since they are only a small proportion of total predictions.

### Recall / Sensitivity
(These tems refer to the same thing, with sensitivity often preferred in medical contexts.)
How many the actual positives did the model correctly identify?

$\text{Recall} = \frac{\text{Number of correct positive predictions}}{\text{Number of positive samples}}$

For two classes:

$\text{Recall} = \frac{TP}{TP + FN}$

Recall helps address the problem described above—the drowning out of false negatives. The system described above, which did well a sample with many negatives but missed proportionally more of the small number of positives, would score worse on recall than on accuracy.

We use recall when the worst thing the model can do is leave out a positive we should have classified.

### Precision
How many of the model's positive predictions were correct?

$\text{Recall} = \frac{\text{Number of correct positive predictions}}{\text{Total number of positive predictions}}$

For two classes:

$\text{Recall} = \frac{TP}{TP + FP}$

Conversely to recall, precision should be our metric of choice when the worst thing the model can do is include a sample that shouldn't be included. The real-world drawbacks of over-inclusion often have to do with volume: if a system that flags possible fraudulent bank transactions for review gets almost every actual fraudulent transaction but forces human review for 50 legitimate transactions for every fraudulent one, there is probably some room for improvement.

### F-1 and F-Beta
It can be useful to have a single value that balances recall's focus on false negatives and precision's focus on false positive. The standard way of doing this by calculate the harmonic mean of precision and recall, also called the F1 score:

$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + text{Recall}}$

The F1 score gives equal importance to precision and recall. However, F1 is only one of a family of F-Beta scores that can vary the weight given to precision vs. recall based on the parameter $\beta$.

$\text{F}_{\beta} = (1+\beta^2) \times \frac{\text{Precision} \times \text{Recall}}{( \beta^2 \times \text{Precision}) + \text{Recall}}$

$\beta > 1$  gives more weight to precision, while $\beta < 1$  gives more weight to recall. F1 score is the version of F-Beta score when $\beta = 1$

### Mean Reciprocal Rank (MRR)
Many classification systems double as ranking systems. Their output is not simply yes/no, positive/negative; they are designed to produce an ordinal output of 'most positive' to 'most negative.' Classification metrics like accuracy, precision, and recall can be applied to a ranked output (more detail below), but it can be useful to zero in on a more detailed measure of how well the system ranks things. One common approach to evaluating a ranked output is mean reciprocal rank (MRR).

Consider a classification system designed to classify items into relevant and irrelevant, such as a search engine or content recommender. The purpose of this system is to provide a user the most relevant items to a given query. In training, let us suppose that relevance is treated as binary: an item is either relevant or irrelevant. The intended output of the system is a ranked list with the items the system scores as most relevant at the top and least relevant at the bottom.

With $J$ inputs of which $U$ are relevant, 

$\mathrm{MRR}\;=\;\frac{1}{U}\sum_{u=1}^{U}\frac{1}{\mathrm{rank}_\mathrm{k}}$

The higher the MRR is, the more highly the system tends to rank true relevant items.

### Applying Classification Metrics to a Ranked Output
It can also be useful to evaluate the traditional classification metrics discussed above in the context of a ranked output. Returning to the bank fraud example from earlier, if one of the system's goal is not to identify every example of fraud but to flag the 100 transactions that are most likely to be fraudulent out of 1000 total transactions, this constrained list of 100 becomes much more important to examine on its own.

Accuracy, precision, and recall can all be calculated for the top $K$ items of a ranked output:

$\mathrm{Accuracy}=\frac{\mathrm{true\ positives\ in\ top\ K}+\mathrm{true\ negatives\ rejected\ to\ produce\ top\ K}}{\mathrm{total\ candidates\ considered\ for\ top\ K}}$

$\mathrm{Recall}=\frac{\mathrm{true\ positives\ in\ all\ top\ K}}{\mathrm{total\ positives}}$

$\mathrm{Precision}=\frac{\mathrm{true\ positives\ in\ all\ top\ K}}{\mathrm{K}}$

## Debugging Tips

### General Debugging Strategy

### Including lots of print() statements will increase visibility of your code and give you insights into what's happening. Especially useful when designing large loops.

### Common Debugging Commands

```python
# Check model summary
print(model)

# Check tensor shapes
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

```
**Q: I’m not sure where my code is breaking, or if it’s hitting all the implementation stages – how can I check the progress of my code?**

Because streaming is incremental, it’s important to provide yourself sanity checks to make sure you’re aware of what stage you’re at. Part of this is the aforementioned code organization technique; in addition, it’s helpful to use print statements at specific moments so you have a sense of what stage your code has reached. For example, if you’re streaming a lot of data, having print statements output at specific intervals so you know how much data has been processed at any given time allows better transparency than would otherwise be granted –

```python
for record in dataset_stream:
                total_checked += 1
                # Progress update
                if total_checked % 10000 == 0: # Note assuming your data is greater than 10,000
                    print(f"  Checked: {total_checked:,}")
```
---

## Best Practices


### Code Organization

Really depends on the individual. A generally adopted approach is to divide your code into meaningful chunks, if e.g. in jupyer notebooks. This helps to distinguish steps from each other and makes it easier to rerun and debug issues. 

### Documentation and Tracking

Keep a simple log of your experiments so you remember what you tried. This can be as simple as comments in your notebook or a text file tracking hyperparameters and results. Save your models with descriptive names that include key info like learning rate and accuracy (e.g., `model_lr001_acc85.pth` instead of `final_model.pth`). If working in a team, add a brief README explaining what your model does, what dataset you used, and how to run the code.

Finally, combine documentation with periodic checkpoints during training. Saving intermediate models lets you recover from crashes, compare different training stages, and experiment with fine-tuning or early stopping without starting from scratch. Good documentation and tracking practices make your workflow more efficient, reliable, and collaborative.

### Performance Optimization

Start simple and optimize only when needed. Common ways to speed up training are to increase the batch size if you have GPU memory available, use mixed precision training with `torch.cuda.amp` to reduce memory usage and speed up computation, and ensure you're using GPU (not CPU) for training. For data loading, use `num_workers=2` and `pin_memory=True` in your DataLoader. If your model is too large for GPU memory, try reducing batch size, using gradient accumulation, or simplifying your architecture. A slightly slower but working model is better than an over-optimized one you can't debug. 

Overall, performance optimization works best when approached gradually: start with a clean, simple pipeline, then optimize only the steps that have the biggest impact.

### Common Pitfalls to Avoid

**Not using `model.eval()` during validation/testing:** This keeps dropout and batch normalization in training mode, giving incorrect results. Always use `model.eval()` before validation and `model.train()` before training.

**Forgetting `torch.no_grad()` during inference:** This wastes memory by tracking gradients you don't need. Wrap validation/test code with `with torch.no_grad():` to save memory.

**Training on unnormalized data:** Your model will struggle to converge. Always better to normalize inputs (e.g., to mean=0, std=1 for images).

**Not shuffling training data:** Set `shuffle=True` in your training DataLoader, otherwise your model might learn patterns based on data order rather than actual features.

**Using test data too early:** Never change your test set until final evaluation. Use it once to report final results - if you tune based on test performance, you're overfitting to the test set.

**Ignoring validation loss:** If training loss decreases but validation loss increases, you're overfitting. Stop training or add regularization (dropout, weight decay).

**Too complex models:** Run some experiments and seek the optimal complexity and depth for your model to prevent overfitting. High complexity is not always (always not) a good thing.

**Changing working environment:** If you need something like a portable SSD to work on multiple devices, remember to make sure the environment is built on the SSD and the drive letter is the same on all devices you would work on.

**Copy-pasting code without understanding:** Take time to understand what each line does. This makes debugging much easier when things break.

## Additional Resources

### Learning Materials
- [Deep Learning Book](https://www.manning.com/books/deep-learning-with-python-second-edition)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Fast.ai Course](https://course.fast.ai/)

### Tools and Libraries
- **PyTorch**: Popular deep learning framework
- **TensorFlow**: Alternative framework
- **Keras**: This is what was mostly used in class

**Last Updated**: December 2025
