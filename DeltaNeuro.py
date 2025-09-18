import numpy as np


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size + 1)

    def predict(self, input_data):
        input_data_bias = np.append(input_data, 1)
        net = np.dot(input_data_bias, self.weights)
        return 1 / (1 + np.exp(-net))

    def train(self, input_data, output_data, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            # print(f"Эпоха {epoch + 1}:")

            for i in range(len(input_data)):
                input_data_bias = np.append(input_data[i], 1)
                prediction = self.predict(input_data[i])
                error = output_data[i] - prediction
                self.weights += learning_rate * error * input_data_bias
                total_error += error ** 2

                # print(f"\tИтерация {i + 1},\tВеса = {self.weights},\tОшибка = {total_error}")

            if total_error < 0.25:
                # print(f"Обучение завершено на эпохе {epoch + 1}")
                return

        # print(f"Достигнут предел эпох, обучение закончено: {epochs}")


def neuro_or():
    input_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_or = np.array([0, 1, 1, 1])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"ИЛИ\"")
    or_neuron = Neuron(2)
    or_neuron.train(input_or, output_or)

    print("=" * 60 + "\nПРОВЕРКА OR")
    for i in range(len(input_or)):
        out = or_neuron.predict(input_or[i])
        print(f"Входные данные: {input_or[i]},\tОжидание: {output_or[i]},\tВывод ИНС: {1 if out > 0.5 else 0}({out})")


def neuro_and():
    input_and = np.array([[0, -1], [0, 1], [1, 0], [1, 1]])
    output_and = np.array([0, 0, 0, 1])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"И\"")
    and_neuron = Neuron(2)
    and_neuron.train(input_and, output_and)

    print("=" * 60 + "\nПРОВЕРКА AND")
    for i in range(len(input_and)):
        out = and_neuron.predict(input_and[i])
        print(f"Входные данные: {input_and[i]},\tОжидание: {output_and[i]},\tВывод ИНС: {1 if out > 0.5 else 0}({out})")


def neuro_not():
    input_not = np.array([[0], [1]])
    output_not = np.array([1, 0])

    print("=" * 60 + "\nЛОГИЧЕСКОЕ \"НЕТ\"")
    not_neuron = Neuron(1)
    not_neuron.train(input_not, output_not)

    print("=" * 60 + "\nПРОВЕРКА NOT")
    for i in range(len(input_not)):
        out = not_neuron.predict(input_not[i])
        print(f"Входные данные: {input_not[i]},\tОжидание: {output_not[i]},\tВывод ИНС: {1 if out > 0.5 else 0}({out})")


def main():
    neuro_or()
    neuro_and()
    neuro_not()


if __name__ == '__main__':
    main()
