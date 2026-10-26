# Обзор готовых моделей для перевода жестового языка

Дата: 2026-10-20

## 1. HuggingFace модели

### opencampus/sign-whisper-german (Немецкий ЖЯ)
- **Архитектура**: Whisper-large-v3 + MediaPipe keypoints
- **Подход**: Извлекает 128 keypoints (руки, верхняя часть тела, лицо) через MediaPipe, подаёт в Whisper как аудио-подобные фичи
- **Язык**: Немецкий жестовый (DGS)
- **Статус**: Рабочая модель, но для немецкого языка
- **Потенциал**: Архитектура подходит для адаптации под русский ЖЯ
- **Ссылка**: https://huggingface.co/opencampus/sign-whisper-german

### star092304/vi-sign-language-videomae-base (Вьетнамский ЖЯ)
- **Архитектура**: VideoMAE (Video Masked Autoencoder)
- **Точность**: Не указана
- **Язык**: Вьетнамский жестовый
- **Основа**: MCG-NJU/videomae-base-finetuned-kinetics
- **Скачиваний**: 314
- **Ссылка**: https://huggingface.co/star092304/vi-sign-language-videomae-base

### huzaifanasirrr/realtime-sign-language-translator (ASL)
- **Архитектура**: ResNet18 + MediaPipe
- **Точность**: 99.60% (изолированные жесты)
- **Язык**: Американский жестовый (ASL)
- **Производительность**: Real-time (25+ FPS)
- **Скачиваний**: 12
- **Ссылка**: https://huggingface.co/huzaifanasirrr/realtime-sign-language-translator

### ihsanahakiim/videomae-base-finetuned-signlanguage
- **Архитектура**: VideoMAE
- **Задача**: Классификация видео жестового языка
- **Скачиваний**: 14-18 (несколько версий)
- **Лицензия**: CC-BY-NC-4.0

## 2. GitHub репозитории

### hukenovs/slovo (Русский ЖЯ) ⭐
- **Датасет**: Slovo (уже используем)
- **Модели**: MViTv2, Swin, ResNet-i3d, SignFlow
- **Лучшая точность**: 64.09% (MViTv2-small-32-2)
- **Формат**: ONNX, TorchScript
- **Язык**: Русский жестовый
- **Ссылка**: https://github.com/hukenovs/slovo

### sign-language-processing/detection-train
- **Модели**: Pre-trained для Python и JavaScript
- **Фреймворк**: Keras, TensorFlow.js
- **Подход**: Transformer для детекции жестов
- **Ссылка**: https://github.com/sign-language-processing/detection-train

### ai-forever/Automated-Sign-Language-Tutor (Русский + Английский)
- **Режимы**: LIVE (real-time), Training mode
- **Языки**: Русский и английский жестовый
- **Особенность**: Dual language support
- **Ссылка**: https://github.com/ai-forever/Automated-Sign-Language-Tutor

### kurshakuz/krsl-recognition (Казахско-Русский ЖЯ)
- **Архитектура**: MediaPipe + TensorFlow
- **Языки**: Казахский и русский жестовый
- **Ссылка**: https://github.com/kurshakuz/krsl-recognition

### hearai/sign-language-review
- **Тип**: Обзор ресурсов
- **Содержание**: Список датасетов, моделей, статей
- **Датасеты**: DGS, PJM, Signor, RWTH-PHOENIX, CSL-Daily, MS-ASL, AUTSL, WLASL
- **Ссылка**: https://github.com/hearai/sign-language-review

## 3. Датасеты для русского ЖЯ

### Slovo (уже используем)
- **Размер**: 19600 окон, 1001 класс (слова + буквы)
- **Источник**: https://www.kaggle.com/datasets/kapitanov/slovo
- **GitHub**: https://github.com/hukenovs/slovo

### Logos (2025) ⭐⭐⭐
- **Размер**: Один из крупнейших датасетов РЖЯ
- **Описание**: "Extensive Russian Sign Language dataset"
- **Статья**: https://arxiv.org/html/2505.10481v1
- **Потенциал**: Может содержать continuous sign language данные

### TheRuSLan
- **Тип**: База данных русского жестового языка
- **Авторы**: Kagirov, Ivanko
- **Ссылка**: https://www.semanticscholar.org/paper/TheRuSLan%3A-Database-of-Russian-Sign-Language-Kagirov-Ivanko/be26be1699ba59f2f00989dec3c6a58d0d1d0465

## 4. Архитектуры для Continuous Sign Language Recognition

### Sign Language Transformers (SLRT)
- **Тип**: Transformer для непрерывного распознавания
- **Статья**: https://www.emergentmind.com/topics/sign-language-recognition-transformer-slrt
- **Состояние**: State-of-the-art для CSLR

### CSLRTransformer (CVPR 2026) ⭐⭐⭐
- **Архитектура**: Pose-only system
- **Датасет**: Isharah
- **Статья**: https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Licciardello_CSLRTransformer_A_Pose-Only_System_for_Continuous_Sign_Language_Recognition_CVPRW_2026_paper.html
- **Особенность**: Только pose keypoints (без RGB)

### SignEdgeLVM (2025)
- **Архитектура**: Transformer
- **Задача**: Continuous Sign Language Recognition and Translation
- **Статья**: https://link.springer.com/article/10.1007/s10791-025-09509-1
- **Цитирований**: 20

## 5. Ключевые выводы

### Что можно использовать прямо сейчас:
1. **Slovo pre-trained модели** (MViTv2) - уже есть датасет, можно скачать веса
2. **sign-whisper-german архитектура** - адаптировать под русский ЖЯ
3. **MediaPipe + Transformer** - стандартный пайплайн

### Что требует доработки:
1. **Continuous recognition** - все найденные модели для isolated gestures
2. **Русский ЖЯ** - большинство моделей для ASL или европейских языков
3. **Sequence-to-sequence** - нужен для непрерывного перевода

### Рекомендуемый путь:
1. Скачать Logos датасет (если доступен)
2. Адаптировать sign-whisper архитектуру под русский ЖЯ
3. Использовать CSLRTransformer для continuous recognition
4. Разметить YouTube курс для получения annotated data

## 6. Следующие шаги

1. ✅ Поиск завершён
2. ⏳ Начать разметку YouTube курса (33 видео)
3. ⏳ Скачать Logos датасет (проверить доступность)
4. ⏳ Адаптировать sign-whisper архитектуру
5. ⏳ Обучить continuous recognition модель
