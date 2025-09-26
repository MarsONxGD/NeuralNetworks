import numpy as np


class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input):
        self.input = input
        input_depth, input_height, input_width = input.shape

        output_height = input_height - self.filter_size + 1
        output_width = input_width - self.filter_size + 1

        self.output = np.zeros((self.num_filters, output_height, output_width))

        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    self.output[f, i, j] = np.sum(
                        input[:, i:i + self.filter_size, j:j + self.filter_size] * self.filters[f]
                    )
        return self.output


class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        num_filters, input_height, input_width = input.shape

        output_height = input_height // self.pool_size
        output_width = input_width // self.pool_size

        self.output = np.zeros((num_filters, output_height, output_width))

        for f in range(num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    self.output[f, i, j] = np.max(
                        input[f,
                        i * self.pool_size:(i + 1) * self.pool_size,
                        j * self.pool_size:(j + 1) * self.pool_size]
                    )
        return self.output


def relu(x):
    return np.maximum(0, x)


def demo_simple_cnn():
    image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0],
        [1, 0, 1, 2, 3],
        [3, 1, 0, 1, 2],
        [2, 3, 1, 0, 1]
    ])

    image = image.reshape(1, 5, 5)

    print("Исходное изображение:")
    print(image[0])
    print()

    conv_layer = ConvLayer(num_filters=2, filter_size=3)
    conv_output = conv_layer.forward(image)

    print("После свертки (первый фильтр):")
    print(conv_output[0])
    print()

    relu_output = relu(conv_output)

    print("После ReLU (первый фильтр):")
    print(relu_output[0])
    print()

    pool_layer = MaxPoolLayer(pool_size=2)
    pool_output = pool_layer.forward(relu_output)

    print("После MaxPooling (первый фильтр):")
    print(pool_output[0])


if __name__ == "__main__":
    demo_simple_cnn()