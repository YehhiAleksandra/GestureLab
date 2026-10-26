#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединение Bukva + YouTube alfavit и переобучение модели.
"""

import sys
from pathlib import Path

import numpy as np


def combine_and_train():
    """Объединить датасеты и переобучить модель."""
    # Загружаем Bukva
    bukva_path = "bukva_processed.npz"
    if not Path(bukva_path).exists():
        print(f"Ошибка: {bukva_path} не найден", file=sys.stderr)
        return

    bukva = np.load(bukva_path)
    bukva_windows = bukva["X"]
    bukva_labels = bukva["names"][bukva["y"]]  # Индексы -> имена букв
    print(f"Bukva: {len(bukva_windows)} окон, {len(set(bukva_labels))} классов")

    # Загружаем YouTube alfavit
    yt_path = "datasets/youtube_ark/alfavit_windows.npz"
    if not Path(yt_path).exists():
        print(f"Ошибка: {yt_path} не найден", file=sys.stderr)
        return

    yt = np.load(yt_path)
    yt_windows = yt["windows"]
    yt_labels = yt["labels"]
    print(f"YouTube alfavit: {len(yt_windows)} окон, {len(set(yt_labels))} классов")

    # Объединяем
    all_windows = np.concatenate([bukva_windows, yt_windows], axis=0)
    all_labels = np.concatenate([bukva_labels, yt_labels], axis=0)
    print(f"Всего: {len(all_windows)} окон, {len(set(all_labels))} классов")

    # Определяем уникальные буквы и создаём маппинги
    unique_labels = sorted(set(all_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}

    # Разделяем на train/val вручную (без sklearn)
    np.random.seed(42)
    indices = np.random.permutation(len(all_windows))
    split_idx = int(len(all_windows) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Сохраняем объединённый датасет в формате train.py
    combined_path = "datasets/combined_bukva_youtube.npz"
    is_val = np.zeros(len(all_windows), dtype=bool)
    is_val[val_idx] = True
    
    np.savez(
        combined_path,
        X=all_windows,
        y=np.array([label_to_idx[l] for l in all_labels]),
        is_val=is_val,
        names=np.array(unique_labels)
    )
    print(f"Сохранено: {combined_path}")

    X_train = all_windows[train_idx]
    X_val = all_windows[val_idx]
    y_train = all_labels[train_idx]
    y_val = all_labels[val_idx]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Импортируем модель
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train import GRU, normalize

    # Создаём модель
    y_train_idx = np.array([label_to_idx[l] for l in y_train])
    y_val_idx = np.array([label_to_idx[l] for l in y_val])

    model = GRU(
        d=126,
        h=128,
        c=len(unique_labels),
        seed=42
    )

    # Обучаем
    print("Обучение модели...")
    model.fit(
        X_train, y_train_idx,
        X_val, y_val_idx,
        epochs=100,
        batch_size=32,
        lr=0.001,
        verbose=True
    )

    # Сохраняем модель
    model_path = "bukva_model_combined.npz"
    model.save(model_path, idx_to_label)
    print(f"Модель сохранена: {model_path}")

    # Оценка
    train_acc = model.evaluate(X_train, y_train_idx)
    val_acc = model.evaluate(X_val, y_val_idx)
    print(f"Train accuracy: {train_acc*100:.1f}%")
    print(f"Val accuracy: {val_acc*100:.1f}%")


if __name__ == "__main__":
    combine_and_train()
