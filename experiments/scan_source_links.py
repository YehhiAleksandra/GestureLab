"""Извлекает внешние ссылки из скопированных манифестов проекта."""
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(r'D:\hfdata\GestureLab_source_manifests')
URL = re.compile(r'https?://[^\s"\'<>)\]`]+')
links = []
for p in ROOT.rglob('*'):
    if not p.is_file() or p.stat().st_size > 50_000_000:
        continue
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        continue
    for m in URL.finditer(text):
        u = m.group(0).rstrip('.,;]}>')
        links.append({'file': str(p.relative_to(ROOT)), 'url': u})
counts = Counter(urlparse(x['url']).netloc.lower().removeprefix('www.') for x in links)
out = {'root': str(ROOT), 'total': len(links), 'domains': dict(counts.most_common()),
       'links': links}
Path('source_links.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                      encoding='utf-8')
print(json.dumps({'total': len(links), 'domains': counts.most_common(20)}, ensure_ascii=False))
print('→ D:\\hfdata\\source_links.json')
