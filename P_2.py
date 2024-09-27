import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function

        self.weights = {
            'input_to_hidden': np.random.randn(self.input_size, self.hidden_size),
            'hidden_to_output':  np.random.randn(self.hidden_size, self.output_size)
        }
        self.biases = {
            'hidden': np.zeros((1, self.hidden_size)), 
            'output': np.zeros((1, self.output_size)) 
        }

    def forward(self, X):
        hidden_input = np.dot(X, self.weights['input_to_hidden']) + self.biases['hidden']
        
        if self.activation_function == 'relu':
            hidden_output = self.relu(hidden_input)
        elif self.activation_function == 'sigmoid':
            hidden_output = self.sigmoid(hidden_input)
        elif self.activation_function == 'tanh':
            hidden_output = self.tanh(hidden_input)
        
        output = np.dot(hidden_output, self.weights['hidden_to_output']) + self.biases['output']
        
        return output
    
    def relu(self,x):
        return np.maximum(0, x)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def tanh(self,x):
        return np.tanh(x)

if __name__ == "__main__":
    
    input_size = 2
    hidden_size = 4
    output_size = 1

    nn_relu = NeuralNetwork(input_size, hidden_size, output_size, activation_function='relu')
    nn_sigmoid = NeuralNetwork(input_size, hidden_size, output_size, activation_function='sigmoid')
    nn_tanh = NeuralNetwork(input_size, hidden_size, output_size, activation_function='tanh')
    
    input_data = np.array([[0,-1], [0,1]])
    
    output_relu = nn_relu.forward(input_data)
    output_sigmoid = nn_sigmoid.forward(input_data)
    output_tanh = nn_tanh.forward(input_data)
    
    print("Output with ReLU activation:")
    print(output_relu)
    
    print("\nOutput with Sigmoid activation:")
    print(output_sigmoid)
    
    print("\nOutput with Tanh activation:")
    print(output_tanh)
