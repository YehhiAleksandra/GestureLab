# Этап 42: интеграция VOGinfo в обучение

Дата: 2026-11-03.

## Цель

Добавить 1637 клипов VOGinfo (89 слов) к 830 клипам SP-10 для преодоления потолка 55%.
Ожидаемое покрытие: (194 + 89) / 353 = 80% фраз бенчмарка достижимы.

## Что сделано

### 1. Расширение головы модели (задача #71) ✓

- Исходная голова: 3993 класса (sp10_expanded/fold*.pt)
- Добавлено: 89 новых слов из VOGinfo
- Новая голова: 4082 класса (sp10_vog_expanded/fold*.pt)
- Все 5 фолдов расширены через `expand_checkpoint`

### 2. Подготовка данных (задача #72) ✓

- Создан комбинированный inventory: `D:/hfdata/sp10_vog_combined.json`
  - SP-10 phrases: 830 клипов
  - VOGinfo clips: 1637 клипов (764 уникальных ID)
  - Всего: 2467 клипов
- Создан комбинированный кэш landmarks: `D:/hfdata/sp10_vog_combined_cache/`
  - 830 файлов из sp10_rsl/landmarks/
  - 764 файла из voginfo_landmarks/

### 3. Конвертация формата VOGinfo (в процессе)

Проблема: VOGinfo landmarks сохранены в формате (N, 54, 3), а обучение ожидает (N, 59, 3).

Решение: скрипт `experiments/convert_voginfo_format.py` переоценивает видео через MediaPipe Holistic
с правильной функцией `frame_of`, сохраняя:
- [0:21] = первая рука (21 точка)
- [21:42] = вторая рука (21 точка)
- [42:49] = pose (7 точек)
- [49:59] = face (10 точек)

Прогресс: ~10 файлов/мин, ~76 мин на 764 файла.
Выход: `D:/hfdata/voginfo_landmarks_fixed/`

### 4. Обучение (задача #73) — следующий шаг

Команда:
```bash
python tools/finetune_ctc_phrases.py \
  --init D:/hfdata/sp10_vog_expanded/fold0.pt \
  --inventory D:/hfdata/sp10_vog_combined.json \
  --cache D:/hfdata/voginfo_landmarks_fixed \
  --work D:/hfdata/sp10_vog_folds \
  --epochs 40 --steps 60 --folds 5 \
  --train-all
```

Ожидаемое время: ~4-6 часов.

### 5. Оценка (задача #74) — после обучения

```bash
python tools/meaning_compare.py \
  --init D:/hfdata/sp10_vog_folds/fold{i}.pt \
  --eval D:/hfdata/sp10_eval.json \
  --lemma
```

Замер:
- Покрытие фраз (ожидаемо: 55% → 70-75%)
- Смысл (meaning score)
- Парный бутстрап против базы (sp10_expanded)

## Ожидаемый результат

- Покрытие бенчмарка: 194/353 → 283/353 (80%)
- Смысл: улучшение на 15-20 п.п. (прогноз)
- Production-модель: sp10_vog_folds/fold*.pt (4082 класса)

## Файлы

- `D:/hfdata/sp10_vog_expanded/` — расширенные чекпойнты (5 фолдов)
- `D:/hfdata/sp10_vog_combined.json` — комбинированный inventory
- `D:/hfdata/voginfo_landmarks_fixed/` — конвертированные landmarks VOGinfo
- `D:/hfdata/sp10_vog_folds/` — обученные модели (после завершения)
