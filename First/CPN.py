import numpy as np
import matplotlib.pyplot as plt


class KohonenMap:
    def __init__(self, grid_size, input_size):
        self.grid_size = grid_size
        self.input_size = input_size
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_size)

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def get_bmu(self, input_vec):
        min_dist = float('inf')
        bmu_idx = (0, 0)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist = self.euclidean_distance(input_vec, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx

    def neighborhood_function(self, distance, radius):
        return np.exp(-distance ** 2 / (2 * (radius ** 2)))

    def train(self, data, epochs=100, initial_lr=0.1, initial_radius=None):
        if initial_radius is None:
            initial_radius = max(self.grid_size) / 2

        for epoch in range(epochs):
            lr = initial_lr * np.exp(-epoch / epochs)
            radius = initial_radius * np.exp(-epoch / epochs)

            for input_vec in data:
                bmu_i, bmu_j = self.get_bmu(input_vec)

                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        neuron_dist = np.sqrt((i - bmu_i) ** 2 + (j - bmu_j) ** 2)

                        if neuron_dist <= radius:
                            influence = self.neighborhood_function(neuron_dist, radius)
                            self.weights[i, j] += lr * influence * (input_vec - self.weights[i, j])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} completed")

    def get_activations(self, input_vec):
        activations = np.zeros(self.grid_size)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                activations[i, j] = np.exp(-self.euclidean_distance(input_vec, self.weights[i, j]))
        return activations.flatten()

    def visualize(self, data=None):
        plt.figure(figsize=(10, 8))

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                plt.scatter(i, j, color=self.weights[i, j], s=500)

        if data is not None:
            for point in data:
                bmu_i, bmu_j = self.get_bmu(point)
                plt.scatter(bmu_i, bmu_j, color=point, edgecolors='black', s=100)

        plt.grid(True)
        plt.title("Kohonen Self-Organizing Map")
        plt.savefig(f'SOM.png', dpi=300, bbox_inches='tight')


class GrossbergLayer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size) * 0.1
        self.bias = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self.sigmoid(np.dot(self.weights, x) + self.bias)

    def train(self, som_activations, targets, learning_rate=0.1):
        outputs = self.forward(som_activations)

        error = targets - outputs

        delta = error * outputs * (1 - outputs)
        self.weights += learning_rate * np.outer(delta, som_activations)
        self.bias += learning_rate * delta

        return np.mean(error ** 2)

    def predict(self, som_activations):
        outputs = self.forward(som_activations)
        return np.argmax(outputs), outputs


def main():
    np.random.seed(1)

    red_colors = np.random.rand(50, 3)
    red_colors[:, 1:] = red_colors[:, 1:] * 0.3

    blue_colors = np.random.rand(50, 3)
    blue_colors[:, 0] = blue_colors[:, 0] * 0.3
    blue_colors[:, 2] = np.clip(blue_colors[:, 2] + 0.3, 0, 1)

    colors = np.vstack([red_colors, blue_colors])
    targets = np.array([0] * 50 + [1] * 50)

    som = KohonenMap(grid_size=(5, 5), input_size=3)
    som.train(colors, epochs=50)

    grossberg = GrossbergLayer(input_size=25, output_size=2)

    print("Обучение слоя Гроссберга...")
    for epoch in range(100):
        total_error = 0
        for i, (color, target) in enumerate(zip(colors, targets)):
            som_activations = som.get_activations(color)

            target_vector = np.zeros(2)
            target_vector[target] = 1

            error = grossberg.train(som_activations, target_vector, learning_rate=0.1)
            total_error += error

        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch + 1}, Средняя ошибка: {total_error / len(colors):.4f}")

    # Тестирование
    print("\nТестирование классификации:")
    correct = 0
    for i, (color, target) in enumerate(zip(colors, targets)):
        som_activations = som.get_activations(color)
        prediction, outputs = grossberg.predict(som_activations)

        if prediction == target:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        if i < 5:
            print(f"Цвет: {color}, Целевой: {target}, Предсказанный: {prediction}, Вероятности: {outputs} {status}")

    accuracy = correct / len(colors) * 100
    print(f"\nТочность классификации: {accuracy:.2f}%")

    som.visualize(colors)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grossberg.weights[0].reshape(5, 5), cmap='viridis')
    plt.title('Веса для класса 0 (красный)')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(grossberg.weights[1].reshape(5, 5), cmap='viridis')
    plt.title('Веса для класса 1 (синий)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('Grossberg_weights.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
