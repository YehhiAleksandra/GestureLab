#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Объединить Bukva + Slovo (только буквы) и обучить модель.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import train_model


# Русская дактильная азбука (33 буквы + no_event)
DACTYL_ALPHABET = set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")


def filter_dactyl_classes(names: np.ndarray, y: np.ndarray, is_val: np.ndarray, X: np.ndarray):
    """Оставить только классы дактильной азбуки."""
    # Найти индексы классов-букв
    letter_indices = []
    letter_names = []
    for i, name in enumerate(names):
        if name == "no_event" or (len(name) == 1 and name in DACTYL_ALPHABET):
            letter_indices.append(i)
            letter_names.append(name)

    print(f"Букв в датасете: {len(letter_names)} из {len(names)} классов")

    # Создать маппинг старых -> новых индексов
    old_to_new = {old: new for new, old in enumerate(letter_indices)}

    # Фильтровать данные
    mask = np.array([y_i in old_to_new for y_i in y])
    X_filtered = X[mask]
    y_filtered = np.array([old_to_new[y_i] for y_i in y[mask]])
    is_val_filtered = is_val[mask]

    print(f"Окон после фильтрации: {len(X_filtered)}")
    return X_filtered, y_filtered, is_val_filtered, np.array(letter_names)


def main():
    # Загрузить Bukva
    print("Загрузка Bukva...")
    bukva = np.load("bukva_processed.npz", allow_pickle=True)
    X_bukva = bukva["X"]
    y_bukva = bukva["y"]
    is_val_bukva = bukva["is_val"]
    names_bukva = bukva["names"]
    print(f"  Bukva: {len(X_bukva)} окон, {len(names_bukva)} классов")

    # Загрузить Slovo
    print("Загрузка Slovo...")
    slovo = np.load("datasets/slovo/slovo_processed.npz", allow_pickle=True)
    X_slovo = slovo["X"]
    y_slovo = slovo["y"]
    is_val_slovo = slovo["is_val"]
    names_slovo = slovo["names"]
    print(f"  Slovo: {len(X_slovo)} окон, {len(names_slovo)} классов")

    # Фильтровать Slovo — только буквы
    print("Фильтрация Slovo (только буквы)...")
    X_slovo_letters, y_slovo_letters, is_val_slovo_letters, names_slovo_letters = filter_dactyl_classes(
        names_slovo, y_slovo, is_val_slovo, X_slovo
    )

    # Проверить что классы совпадают
    print(f"\nBukva классы ({len(names_bukva)}): {sorted(names_bukva)}")
    print(f"Slovo классы ({len(names_slovo_letters)}): {sorted(names_slovo_letters)}")

    # Если классы разные — нужно привести к общему набору
    common_names = sorted(set(names_bukva) & set(names_slovo_letters))
    print(f"\nОбщих классов: {len(common_names)}")

    if len(common_names) < 30:
        print("ВНИМАНИЕ: мало общих классов — возможно проблема с кодировкой")
        return

    # Создать общий маппинг
    name_to_idx = {name: i for i, name in enumerate(common_names)}

    def remap(X, y, is_val, names):
        mask = np.array([n in name_to_idx for n in names[y]])
        X_new = X[mask]
        y_new = np.array([name_to_idx[names[y[i]]] for i in range(len(y)) if mask[i]])
        is_val_new = is_val[mask]
        return X_new, y_new, is_val_new

    X_bukva_r, y_bukva_r, is_val_bukva_r = remap(X_bukva, y_bukva, is_val_bukva, names_bukva)
    X_slovo_r, y_slovo_r, is_val_slovo_r = remap(X_slovo_letters, y_slovo_letters, is_val_slovo_letters, names_slovo_letters)

    print(f"\nПосле маппинга:")
    print(f"  Bukva: {len(X_bukva_r)} окон")
    print(f"  Slovo: {len(X_slovo_r)} окон")

    # Объединить
    X_combined = np.concatenate([X_bukva_r, X_slovo_r], axis=0)
    y_combined = np.concatenate([y_bukva_r, y_slovo_r], axis=0)
    is_val_combined = np.concatenate([is_val_bukva_r, is_val_slovo_r], axis=0)

    print(f"\nОбъединённый датасет:")
    print(f"  Всего окон: {len(X_combined)}")
    print(f"  Классов: {len(common_names)}")
    print(f"  Val доля: {is_val_combined.mean():.1%}")

    # Сохранить
    output_path = Path("combined_dataset.npz")
    np.savez(output_path, X=X_combined, y=y_combined, is_val=is_val_combined, names=np.array(common_names))
    print(f"Сохранено в {output_path}")

    # Обучить модель
    print("\nОбучение модели...")
    res = train_model(X_combined, y_combined, is_val_combined, epochs=100, hidden=128, batch=64, seed=0)
    print(f"\nРезультаты:")
    print(f"  Train top-1: {res['train_top1']:.1f}%  top-3: {res['train_top3']:.1f}%")
    print(f"  Val   top-1: {res['val_top1']:.1f}%  top-3: {res['val_top3']:.1f}%")

    # Сохранить модель
    model_path = Path("combined_model.npz")
    np.savez(model_path, **res["model"].p, names=np.array(common_names), hidden=128)
    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    main()
