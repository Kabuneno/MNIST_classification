# MNIST Digit Classifier

A simple neural network implementation for classifying handwritten digits from the MNIST dataset using only NumPy.

## Overview

This project implements a basic softmax classifier (essentially a single-layer neural network) that can recognize handwritten digits from the MNIST dataset. The implementation is done from scratch using NumPy, with visualization support from Matplotlib and Seaborn.

## Features

- Loads and preprocesses the MNIST dataset
- Implements forward propagation for prediction
- Uses softmax activation and cross-entropy loss
- Implements gradient descent optimization
- Includes batch training with customizable batch size and epochs
- Provides visual evaluation of prediction results

## Dependencies

- NumPy
- TensorFlow (only for loading the MNIST dataset)
- Matplotlib
- Seaborn

## How It Works

### Data Preprocessing

The MNIST dataset is loaded and flattened from 28x28 pixel images to 784-dimensional vectors for processing.

### Model Architecture

The model consists of:
- Input layer (784 neurons corresponding to flattened 28x28 pixel images)
- Output layer (10 neurons for digit classes 0-9)
- Weights matrix (784x10)
- Biases vector (1x10)

### Training Process

1. Initialize weights with small random values and biases with zeros
2. For each epoch:
   - Shuffle the training data
   - Process mini-batches of data
   - Calculate predictions using softmax activation
   - Compute cross-entropy loss
   - Calculate gradients
   - Update weights and biases using gradient descent

### Prediction

The model predicts digits by:
1. Performing forward propagation on input data
2. Applying softmax to get probabilities for each class
3. Selecting the class with the highest probability

## Usage Example

```python
# Train the model
train(X_train_train, y_train, epochs=100, batch_size=32)

# Make predictions
prediction = get_pred(X_test_test[0])
print(f"Predicted digit: {prediction}")

# Evaluate model on test samples
for i in range(100, 110):
    plt.imshow(X_test[i])
    plt.show()
    if get_pred(X_test_test[i]) == y_test[i]:
        print(f"looks like a {get_pred(X_test_test[i])}")
        print("✅")
    else:
        print(f"looks like a {get_pred(X_test_test[i])} but it's {y_test[i]}")
        print("🔴")
```

## Customization

You can adjust the following parameters:
- `learning_rate` in the `update_values` function (default: 0.001)
- `epochs` in the `train` function (default: 100)
- `batch_size` in the `train` function (default: 32)

## Performance

The model achieves basic classification functionality without any hidden layers. For improved accuracy, consider:
- Adding hidden layers
- Implementing regularization
- Using more advanced optimizers like Adam or RMSprop
