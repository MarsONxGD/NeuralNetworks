import numpy as np


class KohonenNeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1

    def distance(self, x):
        return np.linalg.norm(self.weights - x)  # Используем расстояние

    def update_weights(self, x, learning_rate):
        self.weights += learning_rate * (x - self.weights)


class GrossbergNeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.1

    def predict(self, x):
        return np.dot(self.weights, x)

    def update_weights(self, x, target, learning_rate):
        self.weights += learning_rate * (target - self.predict(x)) * x


class CPN:
    def __init__(self, input_size, kohonen_size, output_size):
        self.input_size = input_size
        self.kohonen_size = kohonen_size
        self.output_size = output_size

        self.kohonen_layer = [KohonenNeuron(input_size) for _ in range(kohonen_size)]
        self.grossberg_layer = [GrossbergNeuron(kohonen_size) for _ in range(output_size)]

    def predict(self, x):
        # Находим победителя по минимальному расстоянию
        distances = np.array([neuron.distance(x) for neuron in self.kohonen_layer])
        winner_idx = np.argmin(distances)  # MIN расстояние, а не MAX скалярное произведение

        # Создаем вектор победителя
        winner_vector = np.zeros(self.kohonen_size)
        winner_vector[winner_idx] = 1.0

        # Прямое распространение через слой Гроссберга
        return np.array([neuron.predict(winner_vector) for neuron in self.grossberg_layer])

    def train(self, X, y, kohonen_lr=0.3, grossberg_lr=0.1, epochs=1000):
        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X)):
                x = X[i]
                target = y[i]

                # Фаза 1 - Обучение Кохонена (по расстоянию)
                distances = np.array([neuron.distance(x) for neuron in self.kohonen_layer])
                winner_idx = np.argmin(distances)
                self.kohonen_layer[winner_idx].update_weights(x, kohonen_lr)

                # Фаза 2 - Обучение Гроссберга
                winner_vector = np.zeros(self.kohonen_size)
                winner_vector[winner_idx] = 1.0

                for j, neuron in enumerate(self.grossberg_layer):
                    prediction = neuron.predict(winner_vector)
                    error = target[j] - prediction
                    total_error += abs(error)
                    neuron.update_weights(winner_vector, target[j], grossberg_lr)

            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Ошибка: {total_error:.4f}")

            if total_error < 0.01:
                print(f"Эпоха {epoch}, Ошибка: {total_error:.4f}")
                print("Обучение завершено!")
                break


# Тестовые функции
def test_or():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЛОГИЧЕСКОГО ИЛИ")
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([[-1], [1], [1], [1]])

    cpn = CPN(2, 4, 1)
    cpn.train(X, y, epochs=1000)

    print("\nПРОВЕРКА:")
    for i in range(len(X)):
        prediction = cpn.predict(X[i])
        print(f"Вход: {X[i]}, Ожидание: {y[i][0]}, Предсказание: {prediction[0]:.3f}")


def test_xor():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЛОГИЧЕСКОГО XOR")
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([[-1], [1], [1], [-1]])

    cpn = CPN(2, 4, 1)
    cpn.train(X, y, epochs=1000)

    print("\nПРОВЕРКА:")
    for i in range(len(X)):
        prediction = cpn.predict(X[i])
        print(f"Вход: {X[i]}, Ожидание: {y[i][0]}, Предсказание: {prediction[0]:.3f}")


def test_and():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЛОГИЧЕСКОГО И")
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([[-1], [-1], [-1], [1]])

    cpn = CPN(2, 4, 1)
    cpn.train(X, y, epochs=1000)

    print("\nПРОВЕРКА:")
    for i in range(len(X)):
        prediction = cpn.predict(X[i])
        print(f"Вход: {X[i]}, Ожидание: {y[i][0]}, Предсказание: {prediction[0]:.3f}")


if __name__ == "__main__":
    test_or()
    test_xor()
    test_and()