import numpy as np
from scipy.spatial.distance import cdist


class RBFNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.centers = np.random.randn(hidden_size, input_size)
        self.sigma = np.ones(hidden_size)
        self.weights = np.random.randn(hidden_size, output_size)
        self.bias = np.random.randn(output_size)

    def rbf(self, x, centers, sigma):
        return np.exp(-cdist(x, centers, 'sqeuclidean') / (2 * sigma ** 2))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.hidden_output = self.rbf(x, self.centers, self.sigma)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights) + self.bias)
        return self.output

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        self.centers = X[np.random.choice(X.shape[0], self.centers.shape[0], replace=False)] # K-Mean

        d_max = np.max(cdist(self.centers, self.centers, 'euclidean'))
        self.sigma = np.ones(self.centers.shape[0]) * d_max / np.sqrt(2 * self.centers.shape[0])

        error = None

        for epoch in range(epochs):
            hidden_output = self.rbf(X, self.centers, self.sigma)
            output = self.sigmoid(np.dot(hidden_output, self.weights) + self.bias)

            error = y - output
            delta = error * output * (1 - output)

            self.weights += learning_rate * np.dot(hidden_output.T, delta)
            self.bias += learning_rate * np.sum(delta, axis=0)

        print("=" * 60)
        print(f"Ошибка: {np.mean(np.square(error)):.6f}")


def rbf_and():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [0], [0], [1]])

    and_nn = RBFNeuralNetwork(2, 3, 1)
    and_nn.train(input_data, output_data)

    print("Результаты AND:")
    for i in range(4):
        out = and_nn.forward(input_data[i].reshape(1, -1))
        print(
            f"Входные данные: {input_data[i]},\tОжидание: {output_data[i][0]},\tВывод ИНС: {1 if out > 0.5 else 0}({out[0][0]:.4f})")


def rbf_or():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [1]])

    or_nn = RBFNeuralNetwork(2, 3, 1)
    or_nn.train(input_data, output_data)

    print("Результаты OR:")
    for i in range(4):
        out = or_nn.forward(input_data[i].reshape(1, -1))
        print(
            f"Входные данные: {input_data[i]},\tОжидание: {output_data[i][0]},\tВывод ИНС: {1 if out > 0.5 else 0}({out[0][0]:.4f})")


def rbf_xor():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_data = np.array([[0], [1], [1], [0]])

    or_nn = RBFNeuralNetwork(2, 3, 1)
    or_nn.train(input_data, output_data)

    print("Результаты XOR:")
    for i in range(4):
        out = or_nn.forward(input_data[i].reshape(1, -1))
        print(
            f"Входные данные: {input_data[i]},\tОжидание: {output_data[i][0]},\tВывод ИНС: {1 if out > 0.5 else 0}({out[0][0]:.4f})")


def main():
    rbf_and()
    rbf_or()
    rbf_xor()


if __name__ == "__main__":
    main()
