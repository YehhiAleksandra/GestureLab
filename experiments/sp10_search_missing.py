# -*- coding: utf-8 -*-
"""Поиск 179 отсутствующих слов на spreadthesign.com и скачивание видео."""
import json
import os
import time
import urllib.request
import urllib.parse
import random
from pathlib import Path
from pymorphy3 import MorphAnalyzer

# Конфигурация
MISSING_FILE = Path("reports/evidence/missing_types_sp10.json")
STATE_FILE = Path("D:/hfdata/sp10_rsl/download_state.json")
VIDEO_DIR = Path("D:/hfdata/sp10_rsl/videos")
OUTPUT_FILE = Path("D:/hfdata/sp10_additional_downloads.json")

# Задержки для обхода 429
MIN_DELAY = 2.0
MAX_DELAY = 4.0

# Ротация User-Agent
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]

def lemmatize(word):
    """Лемматизация слова."""
    morph = MorphAnalyzer()
    try:
        return morph.parse(word)[0].normal_form
    except:
        return word

def load_missing_words():
    """Загрузить список отсутствующих слов."""
    data = json.load(open(MISSING_FILE, encoding="utf-8"))
    words = [item["word"].lower() for item in data["classes"]]
    lemmas = list({lemmatize(w) for w in words if w.replace(" ", "").isalpha() and len(w.replace(" ", "")) > 1})
    print(f"Загружено {len(words)} слов, {len(lemmas)} уникальных лемм")
    return lemmas

def search_word_on_spreadthesign(word):
    """Поиск слова на spreadthesign.com, возврат URL видео или None."""
    # URL для поиска (предполагаемый pattern)
    search_url = f"https://www.spreadthesign.com/en.gb/search/?q={urllib.parse.quote(word)}"
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")
            
            # Парсинг HTML для поиска URL видео
            # Предполагаемый pattern: <video src="..."> или <a href="...mp4">
            # Нужно адаптировать под реальную структуру сайта
            import re
            
            # Ищем URL видео (пример pattern, нужно уточнить)
            video_patterns = [
                r'https://media\.spreadthesign\.com/video/mp4/\d+/\d+\.mp4',
                r'data-video-url="([^"]+\.mp4)"',
                r'<source src="([^"]+\.mp4)"',
            ]
            
            for pattern in video_patterns:
                matches = re.findall(pattern, html)
                if matches:
                    return matches[0]
            
            return None
            
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"  429 Too Many Requests для '{word}', ждём дольше...")
            time.sleep(10)
        return None
    except Exception as e:
        print(f"  Ошибка поиска '{word}': {e}")
        return None

def test_single_word(word):
    """Тест поиска одного слова с сохранением HTML для анализа."""
    print(f"\nТест поиска слова: {word}")
    
    search_url = f"https://www.spreadthesign.com/en.gb/search/?q={urllib.parse.quote(word)}"
    print(f"URL: {search_url}")
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode("utf-8")
            
            # Сохранение HTML для анализа
            html_file = Path(f"D:/hfdata/test_search_{word}.html")
            html_file.write_text(html, encoding="utf-8")
            print(f"HTML сохранён: {html_file}")
            print(f"Размер: {len(html)} байт")
            
            # Попытка найти URL видео
            import re
            video_patterns = [
                r'https://media\.spreadthesign\.com/video/mp4/\d+/\d+\.mp4',
                r'data-video-url="([^"]+)"',
                r'<source src="([^"]+\.mp4)"',
                r'video_url["\s:]+["\']([^"\']+)["\']',
            ]
            
            found_urls = []
            for pattern in video_patterns:
                matches = re.findall(pattern, html)
                if matches:
                    found_urls.extend(matches)
                    print(f"  Pattern '{pattern[:30]}...' нашёл: {matches[:3]}")
            
            if found_urls:
                print(f"\nНайдено URL видео: {len(found_urls)}")
                return found_urls[0]
            else:
                print("\nURL видео не найдены. Проверьте HTML вручную.")
                return None
                
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def main():
    """Основная функция."""
    import sys
    
    # Режим теста: python sp10_search_missing.py test <слово>
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        word = sys.argv[2]
        test_single_word(word)
        return
    
    # Основной режим: поиск всех 179 слов
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Загрузка отсутствующих слов
    missing_lemmas = load_missing_words()
    
    # Загрузка существующего состояния
    state = {}
    if STATE_FILE.exists():
        state = json.load(open(STATE_FILE, encoding="utf-8"))
    
    # Результаты поиска
    results = {}
    found = 0
    not_found = 0
    
    print(f"\nПоиск {len(missing_lemmas)} слов на spreadthesign.com...")
    print(f"Задержки: {MIN_DELAY}-{MAX_DELAY} сек между запросами\n")
    
    for i, word in enumerate(missing_lemmas, 1):
        print(f"[{i}/{len(missing_lemmas)}] Поиск: {word}", end=" ")
        
        # Проверка, не найдено ли уже
        if word in results:
            print("(уже найдено)")
            continue
        
        # Поиск слова
        video_url = search_word_on_spreadthesign(word)
        
        if video_url:
            results[word] = {"url": video_url, "status": "found"}
            found += 1
            print(f"✓ найдено")
        else:
            results[word] = {"url": None, "status": "not_found"}
            not_found += 1
            print("✗ не найдено")
        
        # Сохранение промежуточных результатов
        if i % 10 == 0:
            json.dump(results, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  (сохранено промежуточных результатов: {i})")
        
        # Задержка для обхода 429
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
    
    # Финальное сохранение
    json.dump(results, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Поиск завершён:")
    print(f"  Найдено: {found}")
    print(f"  Не найдено: {not_found}")
    print(f"  Результаты сохранены: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    
    # Следующий шаг: скачивание найденных видео
    if found > 0:
        print("Для скачивания найденных видео запустите:")
        print(f"  python experiments/sp10_download_additional.py")

if __name__ == "__main__":
    main()
