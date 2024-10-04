import numpy as np

# a. Single Layer Feed Forward Neural Network

class SingleLayerNN:
    
    def __init__(self, input_size,output_size):
        self.input_size = input_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
    
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            # Compute error (difference between actual and predicted)
            error = y - output
            # Backpropagation step: adjusting weights using gradient descent
            d_output = error * self.sigmoid_derivative(output)
            # Adjust weights and biases
            self.weights += np.dot(X.T, d_output) * learning_rate
            self.bias += np.sum(d_output, axis=0) * learning_rate
            # Print the loss (mean squared error) for monitoring
            loss = np.mean(np.square(error))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Example for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

nn = SingleLayerNN(input_size=2, output_size=1)
# nn.train(X, y, learning_rate=0.1, epochs=10000000)
nn.train(X, y, learning_rate=0.1, epochs=10000)

output = nn.forward(X)

for index, i in enumerate(output):
    print(f"Input: {X[index]}, Prediction: {i[0]:.2f}")
