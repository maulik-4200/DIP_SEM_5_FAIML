import numpy as np

# Multi-Layer Feedforward Neural Network (MLP)

class MultiLayerNN:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Forward pass through the output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute the error (difference between actual and predicted)
            error = y - output
            
            # Backpropagation (output layer)
            d_output = error * self.sigmoid_derivative(output)
            
            # Backpropagation (hidden layer)
            error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_output)
            
            # Update weights and biases (output layer)
            self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
            self.bias_output += np.sum(d_output, axis=0) * learning_rate
            
            # Update weights and biases (hidden layer)
            self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
            self.bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate
            
            # Print the loss (mean squared error) every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Example for XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  

nn = MultiLayerNN(input_size=2, hidden_size=2, output_size=1)

nn.train(X, y, learning_rate=0.1, epochs=10000)

output = nn.forward(X)

for index, prediction in enumerate(output):
    print(f"Input: {X[index]}, Prediction: {prediction[0]:.2f}")
