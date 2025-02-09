import re
import networkx as nx
import matplotlib.pyplot as plt
import pymorphy2
import nltk


nltk.download('stopwords')
from nltk.corpus import stopwords


def text_to_word_graph(text, window_size=2):
    """
    Преобразует текст в граф, где узлами являются слова, а ребра – сосуществование слов в окне заданного размера.

    :param text: Исходный текст (на русском языке)
    :param window_size: Размер окна для учета соседних слов (по умолчанию 2: текущие и следующее слово)
    :return: Граф NetworkX с узлами (словами) и ребрами с атрибутом 'weight'
    """
    words = re.findall(r'\b[а-яё]+\b', text.lower())

    G = nx.Graph()

    for word in set(words):
        G.add_node(word)

    for i in range(len(words)):
        for j in range(i + 1, min(i + window_size, len(words))):
            w1 = words[i]
            w2 = words[j]
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)

    return G


# Пример использования
if __name__ == '__main__':
    text = "Привет, как дела? Привет, что нового в мире технологий? Технологии развиваются стремительно, и мир меняется."

    word_graph = text_to_word_graph(text, window_size=3)

    print("Узлы графа:")
    print(list(word_graph.nodes()))

    print("\nРёбра графа с весами:")
    for u, v, data in word_graph.edges(data=True):
        print(f"{u} - {v}: {data}")

    pos = nx.spring_layout(word_graph, k=0.5, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(word_graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1500,
                     font_size=10)

    edge_labels = nx.get_edge_attributes(word_graph, 'weight')
    nx.draw_networkx_edge_labels(word_graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Граф слов для заданного текста")
    plt.axis('off')
    plt.show()
