"""Копирует только маленькие текстовые манифесты/ссылки из внешней папки проекта."""
import os
import shutil
from pathlib import Path

SRC = Path(r'D:\ДляГлухихAi')
DST = Path(r'D:\hfdata\GestureLab_source_manifests')
EXT = {'.json', '.jsonl', '.txt', '.md', '.csv', '.tsv', '.srt', '.url', '.webloc'}
SKIP = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache'}
MAX_SIZE = 20_000_000
copied = 0
errors = []
for dirpath, dirnames, filenames in os.walk(SRC):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for name in filenames:
        src = Path(dirpath) / name
        if src.suffix.lower() not in EXT:
            continue
        try:
            if src.stat().st_size > MAX_SIZE:
                continue
            rel = src.relative_to(SRC)
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            copied += 1
        except OSError as e:
            errors.append(f'{src}: {e}')
print(f'copied={copied} errors={len(errors)}')
for e in errors[:20]:
    print(e)
print(f'→ {DST}')
