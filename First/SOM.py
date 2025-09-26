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

def main():
    np.random.seed(1)
    colors = np.random.rand(100, 3)

    som = KohonenMap(grid_size=(5, 5), input_size=3)
    som.train(colors, epochs=50)

    som.visualize(colors)

if __name__ == "__main__":
    main()