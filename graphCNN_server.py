import re
import networkx as nx
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
english_stopwords = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Предобработка текста:
      1. Приведение к нижнему регистру.
      2. Токенизация (выделение слов из латинских букв).
      3. Удаление английских стоп-слов.
      4. Лемматизация (приведение к базовой форме) каждого слова.
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [tok for tok in tokens if tok not in english_stopwords]
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return tokens


def build_graph_of_words(tokens, window_size=4):
    """
    Построение графа слов. Узлы - слова, ребро между двумя разными словами,
    если они встречаются в пределах одного окна из window_size слов.
    Вес ребра увеличивается при повторных совместных появлениях.
    """
    G = nx.Graph()
    n = len(tokens)
    for i in range(n):
        word_i = tokens[i]
        if not G.has_node(word_i):
            G.add_node(word_i)
        # Связываем слово с каждым из следующих window_size-1 слов
        window_end = min(i + window_size, n)
        for j in range(i + 1, window_end):
            word_j = tokens[j]
            if word_i == word_j:
                continue
            # Если ребро уже есть, увеличиваем вес, иначе создаем
            if G.has_edge(word_i, word_j):
                G[word_i][word_j]['weight'] += 1
            else:
                G.add_edge(word_i, word_j, weight=1)
    return G


from collections import deque


def extract_subgraph_nodes(G, center, d):
    """
    Извлекает подграф размером d узлов из графа G вокруг узла center.
    BFS обходом находим узлы по уровням, сортируем каждый уровень по степени (убыв.).
    Если узлов меньше d, дополняем "DUMMY".
    """
    visited = {center: 0}
    levels = {}
    queue = deque([(center, 0)])
    while queue:
        node, depth = queue.popleft()
        levels.setdefault(depth, []).append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    # Формируем список узлов, сортируя по уровню и степени
    ordered_nodes = []
    for depth in sorted(levels.keys()):
        # Сортируем узлы уровня: первыми узлы с большей степенью (важные)
        level_nodes = sorted(levels[depth], key=lambda node: G.degree(node), reverse=True)
        ordered_nodes.extend(level_nodes)
        if len(ordered_nodes) >= d:
            break
    ordered_nodes = ordered_nodes[:d]  # берем первые d узлов (если набралось больше)
    # Если узлов меньше d, добавляем фиктивные
    if len(ordered_nodes) < d:
        ordered_nodes += ["DUMMY"] * (d - len(ordered_nodes))
    return ordered_nodes


def subgraph_to_adj_matrix(G, sub_nodes):
    """
    Строит матрицу смежности подграфа (размер d×d).
    Узлы "DUMMY" считаются несвязанными (нулевые строчки/столбцы).
    Вес из исходного графа G, если есть ребро между реальными узлами.
    """
    d = len(sub_nodes)
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if sub_nodes[i] == "DUMMY" or sub_nodes[j] == "DUMMY":
                continue
            if G.has_edge(sub_nodes[i], sub_nodes[j]):
                A[i, j] = G[sub_nodes[i]][sub_nodes[j]].get('weight', 1)
    return A


def get_subgraphs(G, top_N=64, d=5):
    """
    Выбирает top_N узлов с наибольшей степенью и для каждого извлекает подграф из d узлов.
    Возвращает словарь: ключ - центральный узел, значение - словарь {"nodes": [...], "adj": matrix}.
    """
    # Берем узлы по убыванию степени
    nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, deg in nodes_sorted[:top_N]]
    subgraphs = {}
    for center in selected_nodes:
        sub_nodes = extract_subgraph_nodes(G, center, d)
        A = subgraph_to_adj_matrix(G, sub_nodes)
        subgraphs[center] = {"nodes": sub_nodes, "adj": A}
    return subgraphs


from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_transformer = AutoModel.from_pretrained('bert-base-uncased')
model_transformer.eval()


def get_contextual_embedding_matrix(subgraph_nodes, tokenizer, model_transformer, embedding_dim=768):
    text = ' '.join([word for word in subgraph_nodes if word != 'DUMMY'])

    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt')
        outputs = model_transformer(**encoded)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # [seq_len, 768]

        tokens = tokenizer.tokenize(text)
        token_embeddings = hidden_states

        embeddings = []
        current_position = 0

        for word in subgraph_nodes:
            if word == 'DUMMY':
                embeddings.append(np.zeros(embedding_dim))
            else:
                word_tokens = tokenizer.tokenize(word)
                word_len = len(word_tokens)
                # Средний эмбеддинг токенов слова
                word_emb = token_embeddings[current_position:current_position + word_len].mean(dim=0).numpy()
                embeddings.append(word_emb)
                current_position += word_len

    emb_matrix = np.array(embeddings)
    return emb_matrix


import torch
from torch.utils.data import Dataset, DataLoader


embedding_dim=768

class TextGraphDataset(Dataset):
    def __init__(self, texts, label_indices_list, top_N=64, d=5, embedding_dim=768):
        self.texts = list(texts)
        self.labels = list(label_indices_list)
        self.top_N = top_N
        self.d = d
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        true_labels = self.labels[idx]  # список индексов классов
        tokens = preprocess_text(text)
        G = build_graph_of_words(tokens, window_size=4)
        subgraphs = get_subgraphs(G, top_N=self.top_N, d=self.d)

        embedding_matrices = []
        for center, sg in subgraphs.items():
            emb_matrix = get_contextual_embedding_matrix(
                sg["nodes"], tokenizer, model_transformer, self.embedding_dim)
            embedding_matrices.append(emb_matrix)

        if len(embedding_matrices) < self.top_N:
            pad_matrix = np.zeros((self.d, self.embedding_dim))
            while len(embedding_matrices) < self.top_N:
                embedding_matrices.append(pad_matrix)

        if len(embedding_matrices) > self.top_N:
            embedding_matrices = embedding_matrices[:self.top_N]

        X = np.stack(embedding_matrices, axis=0)
        X = X.reshape(self.top_N, self.d * self.embedding_dim)
        X_tensor = torch.tensor(X, dtype=torch.float)

        y = torch.zeros(num_classes, dtype=torch.float)
        for lbl in true_labels:
            y[lbl] = 1.0
        return X_tensor, y


import pandas as pd

df = pd.read_csv("") # Датасет указывать
df_reduced = df.iloc[::4]
df = df_reduced
all_categories = sorted(set(df["l1"]) | set(df["l2"]) | set(df["l3"]))
category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
num_classes = len(all_categories)

# Построим иерархию child->parent для регуляризации
hierarchy = {}
for _, row in df.iterrows():
    l1, l2, l3 = row["l1"], row["l2"], row["l3"]
    idx_l1 = category_to_idx[l1]
    idx_l2 = category_to_idx[l2]
    idx_l3 = category_to_idx[l3]
    # l1 верхнего уровня, нет родителя (можно не добавлять или parent=-1)
    hierarchy[idx_l2] = idx_l1
    hierarchy[idx_l3] = idx_l2
    # Если хотим, можно ввести фиктивный root для всех l1:
    # for each unique l1 that is not yet in hierarchy values, hierarchy[idx_l1] = -1
# добавим parent = -1 для категорий верхнего уровня (необязательно для расчета, функция игнорирует -1)
for cat, idx in category_to_idx.items():
    # если категория не встречается как потомок (т.е. l1)
    if idx not in hierarchy:
        hierarchy[idx] = -1


train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["l3"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["l3"], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_texts = train_df["text"].tolist()
train_labels_indices = []
for _, row in train_df.iterrows():
    train_labels_indices.append([
        category_to_idx[row["l1"]],
        category_to_idx[row["l2"]],
        category_to_idx[row["l3"]]
    ])
val_texts = val_df["text"].tolist()
val_labels_indices = []
for _, row in val_df.iterrows():
    val_labels_indices.append([
        category_to_idx[row["l1"]],
        category_to_idx[row["l2"]],
        category_to_idx[row["l3"]]
    ])
test_texts = test_df["text"].tolist()
test_labels_indices = []
for _, row in test_df.iterrows():
    test_labels_indices.append([
        category_to_idx[row["l1"]],
        category_to_idx[row["l2"]],
        category_to_idx[row["l3"]]
    ])

# Создаем объекты Dataset и DataLoader
train_dataset = TextGraphDataset(train_texts, train_labels_indices, top_N=32, d=5, embedding_dim=embedding_dim)
val_dataset = TextGraphDataset(val_texts, val_labels_indices, top_N=32, d=5, embedding_dim=embedding_dim)
test_dataset = TextGraphDataset(test_texts, test_labels_indices, top_N=32, d=5, embedding_dim=embedding_dim)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F


class GraphCNN(nn.Module):
    def __init__(self, num_classes, graph_size=64, neighbor_size=5, num_channels=50):
        super(GraphCNN, self).__init__()
        self.graph_size = graph_size  # N
        self.neighbor_size = neighbor_size  # d
        self.num_channels = num_channels  # E
        # Входные данные ожидаются как [batch, 1, graph_size, neighbor_size*num_channels]
        # Свертка 1: по оси ширины (узлы подграфа * координаты)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 5), stride=(1, 5))
        self.norm1 = nn.LocalResponseNorm(5, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Свертка 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(1, 5), stride=(1, 1))
        self.norm2 = nn.LocalResponseNorm(5, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Свертка 3
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(1, 5), stride=(1, 1))
        self.norm3 = nn.LocalResponseNorm(5, alpha=0.001 / 9.0, beta=0.75, k=1.0)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # Вычислим размер выхода после сверточных слоев для настройки размеров FC
        dummy_input = torch.zeros(1, 1, graph_size, neighbor_size * num_channels)
        with torch.no_grad():
            dummy_out = self.pool3(self.norm3(F.relu(self.conv3(
                self.pool2(self.norm2(F.relu(self.conv2(
                    self.pool1(self.norm1(F.relu(self.conv1(dummy_input))))))))))))
        conv_output_size = int(np.prod(dummy_out.shape))  # произведение всех размеров кроме batch
        # Полносвязные слои
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch, graph_size, neighbor_size*num_channels]
        x = x.unsqueeze(1)  # добавляем канал: -> [batch, 1, graph_size, neighbor_size*num_channels]
        x = self.pool1(self.norm1(F.relu(self.conv1(x))))
        x = self.pool2(self.norm2(F.relu(self.conv2(x))))
        x = self.pool3(self.norm3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)  # флаттен
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)  # выход без активации
        return logits


# Инициализируем модель
model = GraphCNN(num_classes=num_classes, graph_size=32, neighbor_size=5, num_channels=embedding_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

# Функция потерь и оптимизатор
criterion = nn.BCEWithLogitsLoss()  # мультиклассовая бинарная кросс-энтропия
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)


def compute_dependencies_loss_torch(model, hierarchy, weight=5e-5):
    # Получим веса последнего слоя (num_classes x hidden)
    W = model.module.fc3.weight  # размер [num_classes, hidden_dim]
    reg_loss = 0.0
    # Вычислим средние векторы для всех родительских узлов
    parent_to_children = {}
    for child, parent in hierarchy.items():
        if parent == -1:
            continue
        parent_to_children.setdefault(parent, []).append(child)
    # Для каждого родителя усредним его детей (если родитель тоже child кого-то, можно включить его,
    # но для простоты возьмем только детей, а собственный вес родителя и так будет приближаться к ним через их родителя и т.д.)
    parent_avg = {}
    for parent, children in parent_to_children.items():
        # стеком получим тензор shape [len(children), hidden_dim], усредним по 0-й оси
        child_weights = W[children]  # выбираем веса дочерних классов
        parent_avg[parent] = torch.mean(child_weights, dim=0)
    # Суммируем квадраты разностей
    reg = 0.0
    for child, parent in hierarchy.items():
        if parent == -1:
            continue
        # вектор весов ребенка и родителя (или среднего по детям родителя)
        w_child = W[child]
        if parent in parent_avg:
            w_parent = parent_avg[parent]
        else:
            w_parent = W[parent]
        reg += torch.sum((w_child - w_parent) ** 2)
    return weight * reg


from sklearn.metrics import f1_score

train_losses = []
val_losses = []
train_micro_f1 = []
train_macro_f1 = []
val_micro_f1 = []
val_macro_f1 = []

import tqdm
# Обучение модели
num_epochs = 10
for epoch in tqdm.tqdm(range(1, num_epochs + 1), desc='epoch'):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in tqdm.tqdm(train_loader, desc='batch_train...'):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss_main = criterion(logits, y_batch)
        loss_reg = compute_dependencies_loss_torch(model, hierarchy, weight=5e-5)
        loss = loss_main + loss_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

    # Вычисляем метрики на тренировочном наборе
    model.eval()
    all_train_preds = []
    all_train_targets = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm.tqdm(train_loader, desc='f1_calculating'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.45).float().cpu().numpy()
            targets = y_batch.cpu().numpy()
            all_train_preds.append(preds)
            all_train_targets.append(targets)
    all_train_preds = np.concatenate(all_train_preds, axis=0)
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    train_micro = f1_score(all_train_targets, all_train_preds, average='micro', zero_division=0)
    train_macro = f1_score(all_train_targets, all_train_preds, average='macro', zero_division=0)
    print(f"Epoch {epoch}: f1_micro = {train_micro:.4f}, f1_macro = {train_macro:.4f}")

    # Вычисляем метрики на валидационном наборе
    all_val_preds = []
    all_val_targets = []
    total_val_loss = 0.0
    for X_batch, y_batch in tqdm.tqdm(val_loader, desc='batch_val...'):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_val_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
        targets = y_batch.cpu().numpy()
        all_val_preds.append(preds)
        all_val_targets.append(targets)
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_targets = np.concatenate(all_val_targets, axis=0)
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_micro = f1_score(all_val_targets, all_val_preds, average='micro', zero_division=0)
    val_macro = f1_score(all_val_targets, all_val_preds, average='macro', zero_division=0)

    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)
    train_micro_f1.append(train_micro)
    train_macro_f1.append(train_macro)
    val_micro_f1.append(val_micro)
    val_macro_f1.append(val_macro)

    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    print(f"Train Micro-F1 = {train_micro:.4f}, Train Macro-F1 = {train_macro:.4f}")
    print(f"Val Micro-F1 = {val_micro:.4f}, Val Macro-F1 = {val_macro:.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_micro_f1': train_micro_f1,
        'train_macro_f1': train_macro_f1,
        'val_micro_f1': val_micro_f1,
        'val_macro_f1': val_macro_f1
    }, 'graphcnn_checkpoint.pth')

model.eval()

import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, title="График потерь", save_path=None):
    """
    Отображает или сохраняет график тренировочных и валидационных потерь по эпохам.

    Параметры:
      - train_losses: список потерь на тренировочных данных по эпохам
      - val_losses: список потерь на валидационных данных по эпохам
      - title: заголовок графика (по умолчанию "График потерь")
      - save_path: путь для сохранения изображения в формате PNG.
                   Если None, график отобразится на экране.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'o-', label='Тренировочные потери')
    plt.plot(epochs, val_losses, 's-', label='Валидационные потери')
    plt.title(title)
    plt.xlabel("Эпоха")
    plt.ylabel("Потеря")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"График потерь сохранён как {save_path}")
    else:
        plt.show()


def plot_f1(train_micro, train_macro, val_micro, val_macro, title="Графики F1-метрик", save_path=None):
    """
    Отображает или сохраняет графики F1-метрик (Micro-F1 и Macro-F1) для тренировочного и валидационного наборов.

    Параметры:
      - train_micro: список значений Micro-F1 на тренировке по эпохам
      - train_macro: список значений Macro-F1 на тренировке по эпохам
      - val_micro: список значений Micro-F1 на валидации по эпохам
      - val_macro: список значений Macro-F1 на валидации по эпохам
      - title: общий заголовок для графиков (по умолчанию "Графики F1-метрик")
      - save_path: путь для сохранения изображения в формате PNG.
                   Если None, график отобразится на экране.
    """
    epochs = range(1, len(train_micro) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_micro, 'o-', label='Train Micro-F1')
    plt.plot(epochs, val_micro, 's-', label='Val Micro-F1')
    plt.title("Micro-F1")
    plt.xlabel("Эпоха")
    plt.ylabel("Micro-F1")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_macro, 'o-', label='Train Macro-F1')
    plt.plot(epochs, val_macro, 's-', label='Val Macro-F1')
    plt.title("Macro-F1")
    plt.xlabel("Эпоха")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"График F1-метрик сохранён как {save_path}")
    else:
        plt.show()


plot_loss(train_losses, val_losses, save_path="../loss.png")
plot_f1(train_micro_f1, train_macro_f1, val_micro_f1, val_macro_f1, save_path="f1.png")

