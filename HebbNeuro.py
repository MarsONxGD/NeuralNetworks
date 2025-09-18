import numpy as np


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size + 1) * 0.1

    def predict(self, input_data):
        input_data_bias = np.insert(input_data, 0, 1)
        net = np.dot(input_data_bias, self.weights)
        return 1 if net >= 0 else -1

    def train(self, input_data, output_data, learning_rate=1, epochs=3):
        for epoch in range(epochs):
            # print(f"Эпоха {epoch + 1}:")

            for i in range(len(input_data)):
                input_data_bias = np.insert(input_data[i], 0, 1)
                prediction = self.predict(input_data[i])
                self.weights += learning_rate * input_data_bias * (output_data[i] - prediction)

                # print(f"\tИтерация {i + 1}:")
                # print(f"\t\tВеса = {self.weights}")


def neuro_or():
    input_or = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    output_or = np.array([-1, 1, 1, 1])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"ИЛИ\"")
    or_neuron = Neuron(2)
    or_neuron.train(input_or, output_or)

    print("=" * 60 + "\nПРОВЕРКА OR")
    for i in range(len(input_or)):
        print(f"Входные данные: {input_or[i]},\tОжидание: {output_or[i]},\tВывод ИНС: {or_neuron.predict(input_or[i])}")


def neuro_and():
    input_and = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    output_and = np.array([-1, -1, -1, 1])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"И\"")
    and_neuron = Neuron(2)
    and_neuron.train(input_and, output_and)

    print("=" * 60 + "\nПРОВЕРКА AND")
    for i in range(len(input_and)):
        print(
            f"Входные данные: {input_and[i]},\tОжидание: {output_and[i]},\tВывод ИНС: {and_neuron.predict(input_and[i])}")


def neuro_not():
    input_not = np.array([-1, 1])
    output_not = np.array([1, -1])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"НЕТ\"")
    not_neuron = Neuron(1)
    not_neuron.train(input_not, output_not)

    print("=" * 60 + "\nПРОВЕРКА NOT")
    for i in range(len(input_not)):
        print(
            f"Входные данные: {input_not[i]},\tОжидание: {output_not[i]},\tВывод ИНС: {not_neuron.predict(input_not[i])}")


def main():
    neuro_or()
    neuro_and()
    neuro_not()


if __name__ == '__main__':
    main()
