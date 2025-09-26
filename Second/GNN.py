import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Загрузка данных Cora
dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

print(f"Dataset: {dataset}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Training nodes: {data.train_mask.sum().item()}")
print(f"Validation nodes: {data.val_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")


# ============================================================================
# ПРАКТИКА 7: GCN модель
# ============================================================================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        # Первый GCN слой с агрегацией соседей (усреднение по умолчанию)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Второй GCN слой
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================================================
# ПРАКТИКА 8: GAT модель с механизмом внимания
# ============================================================================

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # Учитываем количество голов внимания для следующего слоя
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.heads = heads

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ============================================================================
# Функции для обучения и оценки
# ============================================================================

def train_model(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def run_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        val_correct = pred[data.val_mask] == data.y[data.val_mask]
        test_correct = pred[data.test_mask] == data.y[data.test_mask]

        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return train_acc, val_acc, test_acc, out


# ============================================================================
# Визуализация эмбеддингов (Практика 8)
# ============================================================================

def visualize_embeddings(model, data, model_name, heads=1):
    """Визуализация эмбеддингов с помощью t-SNE"""
    model.eval()
    with torch.no_grad():
        # Получаем эмбеддинги перед последним слоем
        if model_name == "GCN":
            # Для GCN берем выход после первого слоя
            embeddings = model.conv1(data.x, data.edge_index)
            embeddings = F.relu(embeddings)
        else:  # GAT
            # Для GAT берем выход после первого слоя
            embeddings = model.conv1(data.x, data.edge_index)
            embeddings = F.relu(embeddings)

        embeddings = embeddings.numpy()
        labels = data.y.numpy()

        # Применяем t-SNE для уменьшения размерности до 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Визуализация
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=labels, cmap='tab10', alpha=0.7, s=10)
        plt.colorbar(scatter)
        plt.title(f't-SNE визуализация эмбеддингов ({model_name}, heads={heads})')
        plt.xlabel('t-SNE компонента 1')
        plt.ylabel('t-SNE компонента 2')
        plt.tight_layout()
        plt.savefig(f'embeddings_{model_name}_heads{heads}.png', dpi=300)


# ============================================================================
# Основная функция
# ============================================================================

def main():
    # Эксперименты с разным количеством голов внимания (Практика 8)
    head_configs = [1, 2, 4, 8]

    results = {}

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТЫ С GCN И GAT")
    print("=" * 60)

    # Тестируем GCN модель (Практика 7)
    print("\n--- GCN Model ---")
    gcn_model = GCN(dataset.num_features, 16, dataset.num_classes)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)

    # Обучение GCN
    print("Обучение GCN...")
    for epoch in range(200):
        loss = train_model(gcn_model, data, gcn_optimizer)
        if epoch % 50 == 0:
            train_acc, val_acc, test_acc, _ = run_model(gcn_model, data)
            print(f'Epoch {epoch:3d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    train_acc, val_acc, test_acc, gcn_out = run_model(gcn_model, data)
    results['GCN'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
    print(f'Final GCN - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # Визуализация GCN эмбеддингов
    print("Визуализация GCN эмбеддингов...")
    visualize_embeddings(gcn_model, data, "GCN")

    # Тестируем GAT с разным количеством голов (Практика 8)
    for heads in head_configs:
        print(f"\n--- GAT Model with {heads} heads ---")
        gat_model = GAT(dataset.num_features, 8, dataset.num_classes, heads=heads)
        gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=5e-4)

        # Обучение GAT
        print(f"Обучение GAT с {heads} головами...")
        for epoch in range(200):
            loss = train_model(gat_model, data, gat_optimizer)
            if epoch % 50 == 0:
                train_acc, val_acc, test_acc, _ = run_model(gat_model, data)
                print(f'Epoch {epoch:3d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        train_acc, val_acc, test_acc, gat_out = run_model(gat_model, data)
        results[f'GAT_{heads}heads'] = {'train': train_acc, 'val': val_acc, 'test': test_acc}
        print(f'Final GAT ({heads} heads) - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        # Визуализация GAT эмбеддингов
        print(f"Визуализация GAT эмбеддингов с {heads} головами...")
        visualize_embeddings(gat_model, data, "GAT", heads)

    # Сравнение результатов
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"{'Model':<15} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 50)

    for model_name, accuracies in results.items():
        print(f"{model_name:<15} {accuracies['train']:.4f}     {accuracies['val']:.4f}     {accuracies['test']:.4f}")

    # Анализ механизма агрегации соседей
    print("\n" + "=" * 60)
    print("АНАЛИЗ МЕХАНИЗМОВ АГРЕГАЦИИ")
    print("=" * 60)
    print("GCN: использует усреднение признаков соседних узлов")
    print("GAT: использует взвешенное суммирование с механизмом внимания")
    print("Разное количество голов в GAT позволяет модели фокусироваться на разных аспектах соседних узлов")

    # Анализ влияния количества голов
    print("\n" + "=" * 60)
    print("АНАЛИЗ ВЛИЯНИЯ КОЛИЧЕСТВА ГОЛОВ В GAT")
    print("=" * 60)
    for heads in head_configs:
        key = f'GAT_{heads}heads'
        if key in results:
            print(f"GAT с {heads} головами: Test Accuracy = {results[key]['test']:.4f}")


if __name__ == '__main__':
    main()
