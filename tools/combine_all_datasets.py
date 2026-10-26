#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединение Bukva + Slovo + YouTube для полноценного переводчика.
Slovo содержит 1000 слов/фраз + буквы дактиля.
Bukva и YouTube содержат только буквы.
Объединяем с приоритетом Slovo для букв (больше данных).
"""

import sys
from pathlib import Path

import numpy as np


def combine_all_datasets():
    """Объединить все датасеты."""
    print("=== Объединение датасетов ===\n")
    
    # 1. Slovo (основа: 1000 слов + буквы)
    slovo_path = "datasets/slovo/slovo_processed.npz"
    if not Path(slovo_path).exists():
        print(f"Ошибка: {slovo_path} не найден", file=sys.stderr)
        return
    slovo = np.load(slovo_path, allow_pickle=True)
    slovo_X = slovo["X"]
    slovo_y = slovo["y"]
    slovo_names = slovo["names"]
    print(f"Slovo: {len(slovo_X)} окон, {len(slovo_names)} классов")
    print(f"  Примеры: {', '.join(slovo_names[:10])}")
    
    # 2. Bukva (буквы дактиля)
    bukva_path = "bukva_processed.npz"
    if not Path(bukva_path).exists():
        print(f"Предупреждение: {bukva_path} не найден, пропускаем", file=sys.stderr)
        bukva_X, bukva_y, bukva_names = None, None, None
    else:
        bukva = np.load(bukva_path, allow_pickle=True)
        bukva_X = bukva["X"]
        bukva_names = bukva["names"]
        bukva_y_names = bukva_names[bukva["y"]]
        print(f"\nBukva: {len(bukva_X)} окон, {len(bukva_names)} классов")
        print(f"  Буквы: {', '.join(bukva_names[:10])}")
    
    # 3. YouTube alfavit (буквы дактиля)
    yt_path = "datasets/youtube_ark/alfavit_windows.npz"
    if not Path(yt_path).exists():
        print(f"Предупреждение: {yt_path} не найден, пропускаем", file=sys.stderr)
        yt_X, yt_y_names = None, None
    else:
        yt = np.load(yt_path, allow_pickle=True)
        yt_X = yt["windows"]
        yt_y_names = yt["labels"]
        print(f"\nYouTube: {len(yt_X)} окон, {len(set(yt_y_names))} классов")
    
    # Объединяем: Slovo уже содержит буквы, поэтому Bukva и YouTube добавляем только если
    # их буквы отсутствуют в Slovo
    all_X = [slovo_X]
    all_y_names = [slovo_names[slovo_y]]
    
    # Проверяем, есть ли буквы из Bukva в Slovo
    if bukva_X is not None:
        slovo_letters = set(slovo_names) - {'no_event'}
        bukva_letters = set(bukva_names)
        new_letters = bukva_letters - slovo_letters
        if new_letters:
            print(f"\nДобавляем из Bukva новые буквы: {new_letters}")
            all_X.append(bukva_X)
            all_y_names.append(bukva_y_names)
        else:
            print(f"\nВсе буквы Bukva уже есть в Slovo, пропускаем")
    
    # YouTube буквы
    if yt_X is not None:
        yt_letters = set(yt_y_names)
        slovo_letters = set(slovo_names) - {'no_event'}
        new_letters = yt_letters - slovo_letters
        if new_letters:
            print(f"Добавляем из YouTube новые буквы: {new_letters}")
            all_X.append(yt_X)
            all_y_names.append(yt_y_names)
        else:
            print(f"Все буквы YouTube уже есть в Slovo, пропускаем")
    
    # Объединяем
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y_names, axis=0)
    
    print(f"\n=== Итого ===")
    print(f"Всего окон: {len(X_combined)}")
    print(f"Всего классов: {len(set(y_combined))}")
    
    # Создаём единый словарь классов
    unique_labels = sorted(set(y_combined))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    
    y_idx = np.array([label_to_idx[label] for label in y_combined])
    
    # Разделяем на train/val (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X_combined))
    split_idx = int(len(X_combined) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    is_val = np.zeros(len(X_combined), dtype=bool)
    is_val[val_idx] = True
    
    # Сохраняем
    output_path = "datasets/combined_full_translator.npz"
    np.savez(
        output_path,
        X=X_combined,
        y=y_idx,
        is_val=is_val,
        names=np.array(unique_labels)
    )
    
    print(f"\nСохранено: {output_path}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Статистика по классам
    print(f"\n=== Примеры классов ===")
    print(f"Буквы: {[l for l in unique_labels if len(l) == 1][:10]}")
    print(f"Слова: {[l for l in unique_labels if len(l) > 1][:10]}")
    
    return output_path, unique_labels


if __name__ == "__main__":
    combine_all_datasets()
