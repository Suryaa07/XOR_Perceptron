import numpy as np
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(input_features, hidden_neurons, output_features):
    W1 = np.random.randn(hidden_neurons, input_features)
    W2 = np.random.randn(output_features, hidden_neurons)
    b1 = np.zeros((hidden_neurons, 1))
    b2 = np.zeros((output_features, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, Y, params):
    m = X.shape[1]
    Z1 = np.dot(params["W1"], X) + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = sigmoid(Z2)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs) / m
    return cost, (Z1, A1, params["W1"], params["b1"], Z2, A2, params["W2"], params["b2"]), A2

def backward_propagation(X, Y, cache):
    m = X.shape[1]
    Z1, A1, W1, b1, Z2, A2, W2, b2 = cache
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return {"dZ2": dZ2, "dW2": dW2, "db2": db2, "dZ1": dZ1, "dW1": dW1, "db1": db1}

def update_params(params, gradients, learning_rate):
    params["W1"] -= learning_rate * gradients["dW1"]
    params["W2"] -= learning_rate * gradients["dW2"]
    params["b1"] -= learning_rate * gradients["db1"]
    params["b2"] -= learning_rate * gradients["db2"]
    return params

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # XOR input
Y = np.array([[0, 1, 1, 0]])  # XOR output
hidden_neurons, input_features, output_features = 2, X.shape[0], Y.shape[0]
params = initialize_params(input_features, hidden_neurons, output_features)
epochs, learning_rate = 100000, 0.01
losses = np.zeros((epochs, 1))

for i in range(epochs):
    losses[i, 0], cache, A2 = forward_propagation(X, Y, params)
    gradients = backward_propagation(X, Y, cache)
    params = update_params(params, gradients, learning_rate)

plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

X = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])  # XOR input
cost, _, A2 = forward_propagation(X, Y, params)
prediction = (A2 > 0.5) * 1.0
print(prediction)
