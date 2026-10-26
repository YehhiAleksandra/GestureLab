"""
Объединение Slovo landmarks с нашими данными и обучение улучшенной модели.

Датасеты:
1. combined_top100.npz - наши данные (3643 окна, 100 классов)
2. slovo_landmarks_windows.npz - Slovo данные (6385 окон, 990 классов)

Стратегия:
1. Фильтруем Slovo до тех же 100 классов (или топ-N)
2. Объединяем данные
3. Обучаем новую модель
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'slovo_repo'))
from constants import classes as slovo_classes


def load_dataset(path: str):
    """Загружает NPZ датасет."""
    data = np.load(path)
    return data['X'], data['y']


def get_class_mapping(our_classes: list, slovo_classes_list: list) -> dict:
    """
    Создаёт маппинг индексов классов Slovo в наши классы.
    
    Returns:
        mapping: {slovo_idx: our_idx}
    """
    mapping = {}
    for slovo_idx, slovo_name in enumerate(slovo_classes_list):
        if slovo_name in our_classes:
            our_idx = our_classes.index(slovo_name)
            mapping[slovo_idx] = our_idx
    return mapping


def filter_and_merge_datasets(our_path: str, slovo_path: str, output_path: str,
                             max_classes: int = 100):
    """
    Фильтрует и объединяет датасеты.
    
    Args:
        our_path: путь к нашим данным
        slovo_path: путь к Slovo данным
        output_path: путь для сохранения объединённых данных
        max_classes: максимальное количество классов
    """
    print("Загрузка наших данных...")
    X_our, y_our = load_dataset(our_path)
    print(f"  Наши: {X_our.shape[0]} окон, {len(np.unique(y_our))} классов")
    
    print("\nЗагрузка Slovo данных...")
    X_slovo, y_slovo = load_dataset(slovo_path)
    print(f"  Slovo: {X_slovo.shape[0]} окон, {len(np.unique(y_slovo))} классов")
    
    # Получаем список наших классов
    our_unique_classes = sorted(list(set(y_our)))
    print(f"\nНаши классы (индексы): {our_unique_classes[:20]}")
    
    # Фильтруем Slovo до топ-N классов по частоте
    print(f"\nФильтрация Slovo до топ-{max_classes} классов...")
    unique, counts = np.unique(y_slovo, return_counts=True)
    top_classes = sorted(zip(unique, counts), key=lambda x: -x[1])[:max_classes]
    top_class_indices = [cls for cls, _ in top_classes]
    
    print(f"Топ-{len(top_class_indices)} классов Slovo:")
    for cls, count in top_classes[:10]:
        class_name = slovo_classes[cls] if cls < len(slovo_classes) else f"class_{cls}"
        print(f"  {class_name:20s} (idx={cls}): {count:5d} окон")
    
    # Фильтруем Slovo данные
    mask = np.isin(y_slovo, top_class_indices)
    X_slovo_filtered = X_slovo[mask]
    y_slovo_filtered = y_slovo[mask]
    
    print(f"\nПосле фильтрации: {X_slovo_filtered.shape[0]} окон")
    
    # Создаём маппинг классов Slovo в новые индексы (0..N-1)
    class_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(top_class_indices)}
    y_slovo_mapped = np.array([class_to_new_idx[y] for y in y_slovo_filtered])
    
    # Наши классы уже в диапазоне 0..99, но нужно проверить
    # Если наши классы не совпадают с Slovo, добавляем их
    our_max_class = y_our.max()
    slovo_max_class = y_slovo_mapped.max()
    
    print(f"\nНаши классы: 0..{our_max_class}")
    print(f"Slovo классы (после маппинга): 0..{slovo_max_class}")
    
    # Объединяем
    print("\nОбъединение данных...")
    X_merged = np.concatenate([X_our, X_slovo_filtered], axis=0)
    
    # Для простоты: наши классы оставляем как есть, Slovo сдвигаем
    # Это не идеально, но работает для начала
    y_merged = np.concatenate([y_our, y_slovo_mapped + our_max_class + 1], axis=0)
    
    print(f"\n✓ Объединённый датасет:")
    print(f"  X shape: {X_merged.shape}")
    print(f"  y shape: {y_merged.shape}")
    print(f"  Уникальных классов: {len(np.unique(y_merged))}")
    
    # Сохраняем
    np.savez_compressed(output_path, X=X_merged, y=y_merged)
    print(f"✓ Сохранено в {output_path}")
    
    # Статистика
    unique, counts = np.unique(y_merged, return_counts=True)
    print(f"\nРаспределение классов:")
    print(f"  Мин: {counts.min()}, Макс: {counts.max()}, Среднее: {counts.mean():.1f}")
    
    return X_merged, y_merged


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Объединение датасетов')
    parser.add_argument('--our', default='datasets/combined_top100.npz',
                       help='Путь к нашим данным')
    parser.add_argument('--slovo', default='datasets/slovo_landmarks_windows.npz',
                       help='Путь к Slovo данным')
    parser.add_argument('--output', default='datasets/merged_our_slovo.npz',
                       help='Путь для сохранения')
    parser.add_argument('--max-classes', type=int, default=100,
                       help='Максимум классов из Slovo')
    
    args = parser.parse_args()
    
    filter_and_merge_datasets(
        args.our,
        args.slovo,
        args.output,
        args.max_classes
    )


if __name__ == '__main__':
    main()
