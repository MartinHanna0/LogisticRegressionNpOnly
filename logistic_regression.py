import numpy as np

# Initialize parameters
weights = np.random.rand(X.shape[1], 1) 
bias = 0

# Define Forward Propagation to compute predictions probabilities using sigmoid function
def forward_propagation(x, w, b):
      
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    z = np.matmul(x, w) + b
    y_pred = sigmoid(z)

    return y_pred

# Define Backward Propagation to compute gradients of loss with respect to parameters
def backward_propagation(y, x, w, b):
    
    diff = forward_propagation(x=x, w=w, b=b) - y
    dl_dw = np.dot(x.T , diff) / len(x)
    dl_db = np.mean(diff)
    
    return dl_dw, dl_db

# Define Error to compute an error metric using Binary Cross Entropy formula
def error(y,y_pred):
    m = len(y)
    # To Ensure the Log of 0 is not computed
    epsilon = 1e-15  
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute the loss function
    loss = -np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
    
    # Compute the error (cost) function
    error = loss / m
    return error

# Define Binary Error Metrics, accuracy, recall, precision, and F1 score
def BinaryErrorMetrics(y, y_pred, threshold):
    
    # Convert probability vector to prediction vector according to threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    TP = np.sum((y_pred_binary == 1) & (y == 1))
    FP = np.sum((y_pred_binary == 1) & (y == 0))
    FN = np.sum((y_pred_binary == 0) & (y == 1))
    TN = np.sum((y_pred_binary == 0) & (y == 0))
    
    # Compute metrics
    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    prec = TP / ( TP + FP) if (TP + FP) > 0 else 0
    f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0

    return acc, recall, prec, f1

# Train the model

# Set number of iterations (epochs) and learning rate
epochs = 1000
learning_rate = 1e-2

# Run the training loop, updating parameters, decreasing learning rate periodically, and printing current iterations with error metric to log progress
for i in range(epochs):
    dl_dw, dl_db = backward_propagation(y=y_train, w=weights, b=bias)
    weights, bias = weights - learning_rate * dl_dw , bias - learning_rate * dl_db
    y_pred_train = forward_propagation(x=X_train, w=weights, b=bias)
    error_metric_train = error_fixed(y=y, y_pred=y_pred_train)
    # Printing iteration
    if i % 10 == 9:
        print(f"Epoch {i + 1}, Error: {error_metric_train:.6f}")
    # Reducing learning rate
    if i % 50 == 9:
        learning_rate *= 0.95



# Print confusion matrix (not graphically, but values only) for training set
acc_train , recall_train , prec_train , f1_train = BinaryErrorMetrics(y=y_train, y_pred=y_pred_train, threshold=0.5)

print(f"Train Accuracy: {acc_train:.4f}")
print(f"Train Recall: {recall_train:.4f}")
print(f"Train Precision: {prec_train:.4f}")
print(f"Train F1 Score: {f1_train:.4f}")

# Evaluate model's performance on test set

# Compute probabilities of test set
y_pred_test = forward_propagation(x=X_test, w=weights, b=bias)
# Print out the error metrics to evaluate
acc_test , recall_test , prec_test , f1_test = BinaryErrorMetrics(y=y_test, y_pred=y_pred_test, threshold=0.5)

print(f"Test Accuracy: {acc_test:.4f}")
print(f"Test Recall: {recall_test:.4f}")
print(f"Test Precision: {prec_test:.4f}")
print(f"Test F1 Score: {f1_test:.4f}")
