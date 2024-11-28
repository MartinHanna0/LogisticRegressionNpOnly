# LogisticRegressionNpOnly

This repository contains an implementation of **Logistic Regression** from scratch using only Numpy. It demonstrates key concepts of Logistic Regression, such as forward propagation, backward propagation, loss computation, and performance evaluation with metrics like accuracy, recall, precision, and F1 score.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Logistic Regression is a supervised learning algorithm used for binary classification tasks. In this implementation, all core computations are performed using Numpy without relying on higher-level libraries like Scikit-learn. The training process uses **gradient descent** to minimize the binary cross-entropy loss.

This project serves as a great learning resource for beginners who want to understand the fundamentals of Logistic Regression and its implementation.

---

## Features
- Custom implementation of forward propagation using the **sigmoid function**.
- Backward propagation to compute gradients of weights and bias.
- Training loop with gradient updates using **batch gradient descent**.
- Binary cross-entropy loss for training.
- Performance evaluation using:
  - **Accuracy**
  - **Recall**
  - **Precision**
  - **F1 Score**

---

## Dependencies
- Python 3.6+
- Numpy

Install the required library using pip:
```bash
pip install numpy
```
## Usage

1. **Clone the Repository**:
```bash
   git clone https://github.com/yourusername/logistic-regression-numpy.git
   cd logistic-regression-numpy
```

2. **Prepare Your Dataset**:

Ensure your training and test datasets are stored in Numpy arrays:
- X_train: Training features (shape: (m_train, n_features))
- y_train: Training labels (shape: (m_train, 1))
- X_test: Test features (shape: (m_test, n_features))
- y_test: Test labels (shape: (m_test, 1))

3. **Run the Script**:

python main.py

4. **View the Results**:

- The script will output error metrics (accuracy, recall, precision, and F1 score) and the training progress to the console.

  ## Code Structure

### 1. Initialization
- **Weights (`weights`)**: 
  - Initialized with random values of shape `(n_features, 1)`.
- **Bias (`bias`)**: 
  - Initialized to 0.

---

### 2. Forward Propagation
- **Purpose**: Computes predicted probabilities (`y_pred`) using the sigmoid function.
- **Formula**:
  \[
  y_{\text{pred}} = \frac{1}{1 + e^{-z}}
  \]
  Where:
  \[
  z = X \cdot W + b
  \]

---

### 3. Backward Propagation
- **Purpose**: Computes gradients of the loss with respect to weights (`dl_dw`) and bias (`dl_db`).
- **Formulas**:
  \[
  \frac{\partial L}{\partial W} = \frac{1}{m} X^T \cdot (y_{\text{pred}} - y)
  \]
  \[
  \frac{\partial L}{\partial b} = \frac{1}{m} \sum (y_{\text{pred}} - y)
  \]

---

### 4. Error Function
- **Purpose**: Implements binary cross-entropy loss to evaluate model performance.
- **Formula**:
  \[
  L = -\frac{1}{m} \sum \left( y \cdot \log(y_{\text{pred}}) + (1 - y) \cdot \log(1 - y_{\text{pred}}) \right)
  \]

---

### 5. Performance Metrics
- **Purpose**: Evaluates model predictions using key metrics:
  - **Accuracy**:
    \[
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    \]
  - **Recall**:
    \[
    \text{Recall} = \frac{TP}{TP + FN}
    \]
  - **Precision**:
    \[
    \text{Precision} = \frac{TP}{TP + FP}
    \]
  - **F1 Score**:
    \[
    F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    \]

---

### 6. Training Loop
- **Purpose**: Updates parameters iteratively to minimize the error:
  - Weight update:
    \[
    W = W - \eta \cdot \frac{\partial L}{\partial W}
    \]
  - Bias update:
    \[
    b = b - \eta \cdot \frac{\partial L}{\partial b}
    \]
- **Additional Details**:
  - The loop runs for a predefined number of epochs.
  - Learning rate is periodically decreased to enhance convergence.
  - Training progress and error metrics are logged after every few iterations.

---

## Results

After training, the script outputs the error at regular intervals and evaluates the model's performance on both the training and test datasets. The results include key metrics such as **accuracy**, **recall**, **precision**, and **F1 score**, as well as the confusion matrix for better interpretability.

---

### Training Results
- The training error is logged after every 10 epochs:
  ```plaintext
  Epoch 10, Error: 0.685432
  Epoch 20, Error: 0.657894
  ...
  Epoch 1000, Error: 0.245678

Example final performance metric on train set:
```plaintext
  Train Accuracy: 0.8850
  Train Recall: 0.8902
  Train Precision: 0.8801
  Train F1 Score: 0.8851
```
Example final performance metric on test set:
```plaintext
  Test Accuracy: 0.8700
  Test Recall: 0.8723
  Test Precision: 0.8684
  Test F1 Score: 0.8703
```

