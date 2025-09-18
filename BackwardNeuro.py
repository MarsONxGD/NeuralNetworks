import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.input_hidden = 1
        self.hidden_output = 1
        self.output = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, input_data):
        self.input_hidden = np.dot(input_data, self.weights_input_hidden)
        self.hidden_output = self.sigmoid(self.input_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))
        return self.output

    def backward(self, input_data, output_data, learning_rate):
        output_error = output_data - self.output
        delta_output = output_error * self.sigmoid_derivative(self.output)

        hidden_error = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.weights_input_hidden += input_data.T.dot(delta_hidden) * learning_rate

    def train(self, input_data, output_data, learning_rate=0.2, epochs=1000):
        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, output_data, learning_rate)


def nn_and():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [0], [0], [1]])

    and_nn = NeuralNetwork(2, 4, 1)
    and_nn.train(input_data, output_data)

    print("=" * 60)
    print("Результаты AND:")
    for i in range(4):
        out = and_nn.forward(input_data[i])
        print(
            f"Входные данные: {input_data[i]},\tОжидание: {output_data[i]},\tВывод ИНС: {1 if out > 0.5 else 0}({out})")


def nn_or():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [1]])

    or_nn = NeuralNetwork(2, 4, 1)
    or_nn.train(input_data, output_data)

    print("=" * 60)
    print("Результаты OR:")
    for i in range(4):
        out = or_nn.forward(input_data[i])
        print(
            f"Входные данные: {input_data[i]},\tОжидание: {output_data[i]},\tВывод ИНС: {1 if out > 0.5 else 0}({out})")


def main():
    nn_and()
    nn_or()


if __name__ == "__main__":
    main()
