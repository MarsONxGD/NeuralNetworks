import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score

torch.cuda.empty_cache()

# Загрузка набора данных IMDb
dataset = load_dataset('imdb')

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


# Применение токенизатора ко всему набору данных
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Преобразуем данные в формат PyTorch (важно!)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Подготовка данных
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(7))
test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(7))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Загрузка предварительно обученной модели BERT для классификации
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Обучение
model.train()
for epoch in range(10):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():0.5f}")

# Оценка
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Точность: {accuracy * 100:0.3f}%")
