import numpy as np


class RNNNeuron:
    def __init__(self, input_size, hidden_size):
        # Веса для входных данных
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        # Веса для скрытого состояния
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        # Веса для выхода
        self.W_hy = np.random.randn(hidden_size) * 0.1
        # Смещения
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros(1)

        self.hidden_size = hidden_size
        self.hidden_state = np.zeros((hidden_size, 1))

    def forward(self, x, prev_hidden=None):
        """Прямое распространение"""
        if prev_hidden is not None:
            self.hidden_state = prev_hidden

        # Обновляем скрытое состояние
        self.hidden_state = np.tanh(
            np.dot(self.W_xh, x.reshape(-1, 1)) +
            np.dot(self.W_hh, self.hidden_state) +
            self.b_h
        )

        # Вычисляем выход
        output = np.dot(self.W_hy, self.hidden_state) + self.b_y

        return output[0], self.hidden_state.copy()

    def backward(self, x, hidden_state, output, target, learning_rate):
        """Обратное распространение через время (упрощенное)"""
        # Вычисляем градиенты
        error = output - target

        # Градиенты для выходных весов
        dW_hy = error * hidden_state.T
        db_y = error

        # Градиенты для скрытого состояния
        dh = error * self.W_hy.reshape(-1, 1) * (1 - hidden_state ** 2)

        # Градиенты для входных весов
        dW_xh = np.dot(dh, x.reshape(1, -1))
        dW_hh = np.dot(dh, hidden_state.T)
        db_h = dh

        # Обновляем веса
        self.W_hy -= learning_rate * dW_hy.flatten()
        self.b_y -= learning_rate * db_y
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.b_h -= learning_rate * db_h


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_cell = RNNNeuron(input_size, hidden_size)

    def predict(self, X_sequence):
        """Предсказание для последовательности"""
        predictions = []
        hidden_state = np.zeros((self.hidden_size, 1))

        for x in X_sequence:
            output, hidden_state = self.rnn_cell.forward(x, hidden_state)
            predictions.append(output)

        return predictions, hidden_state

    def train(self, X_sequences, y_sequences, learning_rate=0.01, epochs=1000):
        """Обучение сети"""
        for epoch in range(epochs):
            total_error = 0

            for seq_idx in range(len(X_sequences)):
                X_seq = X_sequences[seq_idx]
                y_seq = y_sequences[seq_idx]

                # Прямое распространение
                predictions, final_hidden = self.predict(X_seq)

                # Обратное распространение (упрощенное - только последний шаг)
                if len(X_seq) > 0:
                    # Обучаем на последнем элементе последовательности
                    x_last = X_seq[-1]
                    hidden_state = final_hidden
                    output = predictions[-1]
                    target = y_seq[-1] if isinstance(y_seq, list) else y_seq

                    self.rnn_cell.backward(
                        x_last, hidden_state, output, target, learning_rate
                    )

                    total_error += abs(output - target)

            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Ошибка: {total_error:.4f}")

            if total_error < 0.01:
                print(f"Эпоха {epoch}, Ошибка: {total_error:.4f}")
                print("Обучение завершено!")
                break


# Тестовые функции для RNN
def test_rnn_sequence_prediction():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ RNN - ПРЕДСКАЗАНИЕ ПОСЛЕДОВАТЕЛЬНОСТИ")

    # Простая последовательность: предсказание следующего числа
    X_sequences = [
        [np.array([1.0]), np.array([2.0]), np.array([3.0])],
        [np.array([2.0]), np.array([3.0]), np.array([4.0])],
        [np.array([3.0]), np.array([4.0]), np.array([5.0])]
    ]
    y_sequences = [4.0, 5.0, 6.0]  # Следующее число после последовательности

    rnn = RNN(1, 4, 1)
    rnn.train(X_sequences, y_sequences, epochs=500)

    print("\nПРОВЕРКА:")
    test_sequences = [
        [np.array([4.0]), np.array([5.0]), np.array([6.0])],
        [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    ]

    for i, seq in enumerate(test_sequences):
        predictions, _ = rnn.predict(seq)
        print(f"Последние значения: ",end="")
        for x in seq:
            print(f"{x[0]:.3f}",end="; ")
        print(f"\nПредсказание следующего числа: {predictions[-1]:.3f}")
        print(f"Ожидание: {seq[-1][0] + 1 if i == 0 else 4.0}")
        print()


def test_rnn_binary_sequence():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ RNN - БИНАРНЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ")

    # Распознавание паттернов в бинарных последовательностях
    X_sequences = [
        [np.array([0, 1]), np.array([1, 0]), np.array([0, 1])],  # Паттерн: 0-1, 1-0, 0-1
        [np.array([1, 0]), np.array([0, 1]), np.array([1, 0])],  # Паттерн: 1-0, 0-1, 1-0
    ]
    y_sequences = [1.0, -1.0]  # Метки для разных паттернов

    rnn = RNN(2, 8, 1)
    rnn.train(X_sequences, y_sequences, epochs=2000)

    print("\nПРОВЕРКА:")
    test_seqs = [
        [np.array([0, 1]), np.array([1, 0]), np.array([0, 1])],  # Должен вернуть ~1
        [np.array([1, 0]), np.array([0, 1]), np.array([1, 0])],  # Должен вернуть ~-1
    ]

    for i, seq in enumerate(test_seqs):
        predictions, _ = rnn.predict(seq)
        print(f"Последние значения: ",end="")
        for x in seq:
            print(f"{x[0]:.3f}",end="; ")
        print(f"\nПредсказание: {predictions[-1]:.3f}")
        print(f"Ожидание: {1.0 if i == 0 else -1.0}")
        print()


def test_rnn_sine_wave():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ RNN - ПРЕДСКАЗАНИЕ СИНУСОИДЫ")

    # Генерация данных синусоиды
    time_steps = 10
    x = np.linspace(0, 4 * np.pi, 50)
    y = np.sin(x)

    X_sequences = []
    y_sequences = []

    for i in range(len(y) - time_steps):
        X_sequences.insert(0, [np.array([y[i + j]]) for j in range(time_steps)])
        y_sequences.insert(0, y[i + time_steps])

    rnn = RNN(1, 10, 1)
    rnn.train(X_sequences, y_sequences, learning_rate=0.01, epochs=1000)

    print("\nПРОВЕРКА:")
    # Тест на последней последовательности
    test_seq = X_sequences[-1]
    predictions, _ = rnn.predict(test_seq)
    print(f"Последние значения: ",end="")
    for x in test_seq:
        print(f"{x[0]:.3f}",end="; ")
    print(f"\nПредсказание следующего: {predictions[-1]:.3f}")
    print(f"Реальное следующее значение: {y_sequences[-1]:.3f}")


if __name__ == "__main__":
    test_rnn_sequence_prediction()
    test_rnn_binary_sequence()
    test_rnn_sine_wave()