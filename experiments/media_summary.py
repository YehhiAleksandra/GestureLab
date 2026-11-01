"""Краткая сводка media_inventory.json."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

inv = json.loads(Path('media_inventory.json').read_text(encoding='utf-8'))
by_dir = defaultdict(lambda: [0, 0])
ext = Counter()
for x in inv['media']:
    parts = Path(x['rel']).parts
    top = parts[0] if len(parts) > 1 else '(корень)'
    by_dir[top][0] += 1
    by_dir[top][1] += x['size']
    ext[Path(x['rel']).suffix.lower()] += 1
print('Медиа по папкам:')
for k, (n, s) in sorted(by_dir.items(), key=lambda kv: -kv[1][1]):
    print(f'  {k:24} {n:4} файла, {s / 1e6:9.1f} МБ')
print('Типы:', dict(ext))
print('\nКрупнейшие медиа:')
for x in sorted(inv['media'], key=lambda r: -r['size'])[:20]:
    print(f"  {x['size'] / 1e6:8.1f} МБ  {x['rel']}")
print('\nСсылки по доменам:')
dom = Counter(urlparse(x['url']).netloc.lower().removeprefix('www.') for x in inv['links'])
for d, n in dom.most_common():
    print(f'  {d:30} {n}')
print('\nСсылки:')
for x in inv['links']:
    print(f"  [{Path(x['file']).name}] {x['url'][:160]}")
