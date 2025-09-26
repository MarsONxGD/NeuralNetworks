import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Загрузка данных Cora
dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]


# Определение GNN-модели
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Инициализация модели, оптимизатора и функции потерь
model = GNN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


# Обучение модели
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1)
    correct = pred[data.train_mask] == data.y[data.train_mask]
    acc = int(correct.sum()) / int(data.train_mask.sum())
    return loss.item(), acc


# Оценка точности модели
def eval():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct_test = pred[data.test_mask] == data.y[data.test_mask]
    correct_val = pred[data.val_mask] == data.y[data.val_mask]
    test_acc = int(correct_test.sum()) / int(data.test_mask.sum())
    val_acc = int(correct_val.sum()) / int(data.val_mask.sum())
    return test_acc, val_acc


def main():
    # Запуск обучения и тестирования
    for epoch in range(500):
        loss, train_acc = train()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')

    test_acc, val_acc = eval()
    print(f'Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}')


if __name__ == '__main__':
    main()
