# DeepGraphCNN
Алгоритм, который использует глубокую свёрточную нейронную сеть, адаптированную для графовых структур (Graph-CNN), чтобы преобразовывать текст в представление, способное уловить не только последовательные (локальные) связи между словами, но и более дальние, не последовательные зависимости. При этом для учета иерархической структуры меток (например, когда один класс является обобщающим для нескольких более мелких) вводится механизм рекурсивной регуляризации, который заставляет параметры классификаторов для родительских и дочерних классов быть похожими.

Разработка разбита на несколько этапов:

- Преобразование текста в графы (Done)
- Выделение подграфов и нормализация (Done)
- Представление узлов в виде эмбеддингов
- Deep Graph-CNN
- Рекурсивная регуляризация
