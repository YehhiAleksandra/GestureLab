"""Инвентаризация видео, датасетов и ссылок в D:\\hfdata без рекурсии в огромные кэши."""
import json
import os
import re
from pathlib import Path

ROOT = Path(r'D:\hfdata')
MEDIA = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.wav', '.mp3'}
DATA = {'.npz', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.srt'}
URL = re.compile(r'https?://[^\s"\'<>]+')
SKIP = {'pip-cache', 'uv-cache', 'hf-cache', '__pycache__', '.git', '.pytest_cache', '.ruff_cache'}
MAX_DEPTH = 4

inventory = {'root': str(ROOT), 'media': [], 'datasets': [], 'url_files': [], 'totals': {}}
url_files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = Path(dirpath).relative_to(ROOT)
    depth = len(rel.parts)
    dirnames[:] = [d for d in dirnames if d not in SKIP and depth < MAX_DEPTH]
    for name in filenames:
        p = Path(dirpath) / name
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        item = {'path': str(p), 'rel': str(p.relative_to(ROOT)), 'size': size,
                'mtime': p.stat().st_mtime}
        if ext in MEDIA:
            inventory['media'].append(item)
        elif ext in DATA:
            if size < 100_000_000:
                inventory['datasets'].append(item)
        if ext in {'.txt', '.json', '.jsonl', '.csv', '.md'} and size < 20_000_000:
            url_files.append(p)

links = []
for p in url_files:
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        continue
    for m in URL.finditer(text):
        u = m.group(0).rstrip(').,;]}>')
        links.append({'file': str(p), 'url': u})
inventory['url_files'] = [{'path': str(p), 'size': p.stat().st_size} for p in url_files]
inventory['links'] = links
inventory['totals'] = {
    'media_files': len(inventory['media']),
    'media_bytes': sum(x['size'] for x in inventory['media']),
    'data_files': len(inventory['datasets']),
    'data_bytes': sum(x['size'] for x in inventory['datasets']),
    'links': len(links),
}
out = ROOT / 'media_inventory.json'
out.write_text(json.dumps(inventory, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(inventory['totals'], ensure_ascii=False))
print(f'→ {out}')
