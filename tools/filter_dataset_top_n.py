#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фильтрация датасета: оставляем только топ-N классов для лучшего качества.
"""

import sys
from pathlib import Path

import numpy as np


def filter_dataset(input_path: str, output_path: str, top_n: int = 100):
    """Оставить только top_n самых частых классов."""
    print(f"Загрузка: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    names = data["names"]
    is_val = data["is_val"]
    
    print(f"Всего: {len(X)} окон, {len(names)} классов")
    
    # Считаем частоту
    unique, counts = np.unique(y, return_counts=True)
    
    # Берём top_n
    sorted_idx = np.argsort(-counts)[:top_n]
    kept_classes = unique[sorted_idx]
    
    print(f"Оставляем {len(kept_classes)} классов")
    print(f"  Минимум примеров: {counts[sorted_idx[-1]]}")
    print(f"  Максимум примеров: {counts[sorted_idx[0]]}")
    
    # Фильтруем данные
    mask = np.isin(y, kept_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    is_val_filtered = is_val[mask]
    
    print(f"После фильтрации: {len(X_filtered)} окон")
    
    # Перенумеровываем классы
    new_names = names[kept_classes]
    old_to_new = {old: new for new, old in enumerate(kept_classes)}
    y_new = np.array([old_to_new[yi] for yi in y_filtered])
    
    # Сохраняем
    np.savez(
        output_path,
        X=X_filtered,
        y=y_new,
        is_val=is_val_filtered,
        names=new_names
    )
    
    print(f"\nСохранено: {output_path}")
    print(f"  Окон: {len(X_filtered)}")
    print(f"  Классов: {len(new_names)}")
    
    # Статистика
    print(f"\nПримеры классов:")
    for i, name in enumerate(new_names[:10]):
        count = sum(y_new == i)
        print(f"  {name}: {count}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/combined_full_translator.npz"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/combined_top100.npz"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    filter_dataset(input_path, output_path, top_n)
