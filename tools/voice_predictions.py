#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Озвучка предсказанных слов через edge-tts.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import edge_tts


async def voice_text(text: str, output_path: str, voice: str = "ru-RU-SvetlanaNeural"):
    """Озвучить текст."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Озвучка предсказанных слов")
    parser.add_argument("predictions_json", help="Путь к JSON с предсказаниями")
    parser.add_argument("--output", default="voiced_output.mp3", help="Выходной аудиофайл")
    parser.add_argument("--voice", default="ru-RU-SvetlanaNeural", help="Голос TTS")
    args = parser.parse_args()
    
    # Загружаем предсказания
    with open(args.predictions_json, encoding="utf-8") as f:
        data = json.load(f)
    
    text = data.get("text", "")
    if not text:
        print("Нет текста для озвучки", file=sys.stderr)
        return
    
    print(f"Текст для озвучки ({len(text)} символов):")
    print(text[:200] + "..." if len(text) > 200 else text)
    print()
    
    # Озвучиваем
    print(f"Озвучиваю голосом {args.voice}...")
    asyncio.run(voice_text(text, args.output, args.voice))
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
