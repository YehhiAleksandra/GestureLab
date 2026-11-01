# Правила проекта GestureLab PC

Проект: этап 0 ТЗ «Переводчик жестов» — измерить, в каком положении на теле камера
видит обе кисти в ≥90% кадров. Две реализации пишут совместимый JSON: PC (этот репо)
и Android (GestureLab v0.1, вне репо).

## Окружение (не менять без причины)

- venv **только** `D:\venvs\gesturelab` — ASCII-путь обязателен: C++-часть MediaPipe
  открывает графы как ANSI-пути и падает на кириллице (B22). `D:\ДляГлухихAi\...` для
  venv не годится, сам репозиторий может лежать где угодно.
- Python 3.11.9, `mediapipe==0.10.14` (точный пин: legacy `mp.solutions` удалён из
  сборок ≥0.10.30 — B21), opencv-python 5.0.0, numpy BSD-3-Clause.
- В Git Bash всегда `PYTHONIOENCODING=utf-8`, иначе вывод кириллицей падает в cp1251.
- Пути для Windows-программ — в формате `C:\...`: shell-вид `/c/...` или `/tmp/...` для
  `cv2.VideoCapture` не существует (B26), `open_video` об этом прямо говорит.
- `cv2.imwrite`/`imread` на кириллическом пути не работают — в `tools/` читают и пишут
  через `imdecode`/`imencode` + `tofile`.

## Гейт перед коммитом и пушем

```
PYTHONIOENCODING=utf-8 /d/venvs/gesturelab/Scripts/python.exe -m pytest -q   # 40 passed
```

- Тесты — часть контракта: новый баг сначала воспроизводится тестом, потом чинится.
  `xfail` не ставить надолго: в фазах 1–3 он был механическим трекером B1/B3 и снят,
  когда баги закрыты.
- Изменили вывод `summarize.py` → `python tools/regen_golden.py` и **читать `git diff
  tests/golden`**: в диффе должны быть только ожидаемые строки.
- `git push` блокируется хуком `cicd-gate`, пока в сессии не прогнан зелёный гейт; CI —
  `.github/workflows/ci.yml` (ubuntu, Python 3.11, фикстуры + pytest). Обход `--no-verify`
  запрещён. Репозиторий приватный (`YehhiAleksandra/GestureLab`).

## Формат данных

- Схема сессии **v2** (`kind`, `summary.recorded_frames`); v1 от Android и старых записей
  читается через `kind_of()` / `recorded_frames_of()`. Молчаливое изменение формата
  запрещено: только версия + миграция + тест на миграцию.
- Запись сессии атомарная и с валидацией схемы (`save_session`): битый JSON не может
  появиться в `sessions/`.
- `sessions/` и `reports/*.csv` в `.gitignore`.

## Приватность

- Видео с камеры **не сохраняется** намеренно — только точки ландмарок в JSON.
- В репозиторий не класть кадры реальных людей. Скриншоты в `reports/img/` — синтетические.
- Данные записей исполнителей никуда не отправлять (внешние сервисы, API, облако).

## Инструменты

| Команда | Что делает |
| --- | --- |
| `python make_fixture.py` | 8 детерминированных фикстур JSON, включая граничные |
| `python tools/regen_golden.py` | пересборка золотых файлов сводки |
| `python tools/demo_run.py` | виртуальный прогон без камеры: PNG состояний HUD в `reports/img/` |
| `python tools/demo_run.py --video` | + синтетическое видео и настоящий offline-прогон через MediaPipe |
| `python tools/demo_run.py --image фото.jpg` | детекция на одном кадре (своё фото, не коммитить) |
| `python gesturelab.py --calibrate sessions` | статистика положений по записям |
| `python summarize.py sessions --by-gesture` | сводка; «не жест» по умолчанию отдельно |
| `python tools/sweep_decode.py --work D:/hfdata/phrase_folds` | перебор декодера на готовых фолдах без обучения (LM, word_bonus, beam) |
| `python tools/finetune_ctc_phrases.py --train-all` | production-чекпойнт на всех фразовых клипах (CV-метрик у него нет, B54) |
| `python tools/centroid_eval.py --chunk 32 --topk 5` | one-shot центроиды по эмбеддингам энкодера: 893 клипа словаря → recall на фразах без обучения (этап 21) |
| `python tools/rerank_eval.py` | фьюжн: n-best цепочки фолдов + центроидная близость, перебор λ офлайн (этап 21 §7) |
| `python tools/finetune_dict.py` | словарная адаптация: голова + энкодер на 893 клипах канала, 5 фолдов по фразам (этап 21 §9) |
| `python tools/fusion_translate.py` | LLM-перевод с судьёй: production-контекст против контекста с центроидами ($≈0.02/прогон) |

## Решения, которые за Вадимом (не принимать самому)

- **B2 / пин OpenCV**: замер 2026-10-19 (`reports/img/b2_fonts.png`) — на 5.0.0 кириллицу
  рисуют все пять шрифтов Hershey, на 4.12 были «?????». Лечится одной строкой в
  `requirements.txt`, кода не требует — но это смена поведения превью у других машин.
- Схема v3: `handedness` (какая рука) и provenance-поля.
- Третье положение «подставка» в протоколе этапа 0.
- Миграция с legacy `mp.solutions` на MediaPipe Tasks `HandLandmarker`.
- Внешние датасеты жестового языка с лицензией CC BY-SA (Slovo/easy_sign) — совместимость
  с закрытым репо и приватностью.

## Проверка утверждений о коде

После правок Python — NoWreck Claims Mode: срез HEAD против рабочего дерева, **претензии
передаются JSON-строкой, а не путём к файлу** (`--claims "$(cat файл)"` — иначе
`total_claims: 0`):

```bash
S=$(mktemp -d) && mkdir -p $S/gl_pre $S/gl_post
git archive HEAD | tar -x -C $S/gl_pre
git ls-files -z | tar --null -T - -cf - | tar -x -C $S/gl_post   # только отслеживаемые файлы:
# datasets/ вырос до десятков ГБ, tar всего дерева копирует 50 ГБ игнорируемых данных
nowreck fix --pre $S/gl_pre --post $S/gl_post \
  --claims "$(cat reports/evidence/nowreck_claims_phase5.json)" --json
```

Претензии лежат в `reports/evidence/`: фаза 4 — 45/45 подтверждено, виртуальный прогон —
23/23. Ключ в JSON — `type`, путь относительный к корню снапшота; межмодульные вызовы
всегда UNVERIFIABLE.

## Отчётность

Каждая фаза — файл `reports/NN_<фаза>.md`: факты со ссылками и датой проверки, баги с
тяжестью, что не сделано и почему, следующие шаги. Итог — `reports/05_summary.md`,
виртуальный прогон — `reports/06_virtual_run.md`.
