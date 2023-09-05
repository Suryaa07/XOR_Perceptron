# XOR_Perceptron
Title: Solving XOR Problem using Multilayer Perceptron

## Overview
This repository contains Python code to solve the XOR problem using a Multilayer Perceptron (MLP). The XOR problem is a classic problem in machine learning that involves creating a neural network to predict XOR outputs.

## Dependencies
- NumPy: A library for numerical operations in Python.
- Matplotlib: A plotting library for visualizing the loss during training.

## Code Structure
The code is organized as follows:

### Activation Function
- `sigmoid(z)`: Defines the sigmoid activation function used in the neural network.

### Initialization
- `initialize_params(input_features, hidden_neurons, output_features)`: Initializes the neural network parameters (weights and biases).

### Forward Propagation
- `forward_propagation(X, Y, params)`: Performs forward propagation through the neural network and computes the cost.

### Backward Propagation
- `backward_propagation(X, Y, cache)`: Computes gradients using backward propagation.

### Parameter Update
- `update_params(params, gradients, learning_rate)`: Updates the neural network parameters using gradient descent.

### XOR Data and Training
- Defines the XOR input (`X`) and output (`Y`) data.
- Initializes the neural network parameters.
- Specifies the number of training epochs and learning rate.
- Iterates through training epochs, updating parameters and recording the loss.

### Visualization
- Plots the loss values over training epochs to visualize the learning progress.

### Testing
- Evaluates the trained model on a new set of XOR inputs.

## Usage
1. Install the required dependencies: NumPy and Matplotlib.
2. Run the provided code to train the MLP on the XOR problem.
3. The loss values over epochs will be displayed as a plot.
4. Test the trained model on new XOR inputs to see the predictions.

## Results
The code demonstrates how to use a Multilayer Perceptron to solve the XOR problem. After training, the model can predict XOR outputs accurately.

Feel free to modify the hyperparameters, network architecture, or other aspects of the code to experiment with different configurations and datasets.

For more details, please refer to the code and comments in the repository.
