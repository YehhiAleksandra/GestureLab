"""Речь видео -> reports/ref/<имя>.json с таймкодами слов (локально, faster-whisper, CPU)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

OUT = Path("reports/ref")


def transcribe(model: WhisperModel, video: Path) -> None:
    out = OUT / f"{video.stem}.json"
    if out.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        segs, _ = model.transcribe(str(wav), language="ru", vad_filter=True, word_timestamps=True)
        data = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                 "words": [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for w in s.words]}
                for s in segs]
    tmp_out = out.with_suffix(".tmp")
    tmp_out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp_out.replace(out)
    print(f"{video.name}: {len(data)} фраз", flush=True)


def main(paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8")
    for p in paths:
        p = Path(p)
        for v in sorted(p.glob("*.mp4")) if p.is_dir() else [p]:
            transcribe(model, v)


if __name__ == "__main__":
    main(sys.argv[1:])
