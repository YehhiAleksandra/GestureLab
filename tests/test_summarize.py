import csv
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_summarize(repo_python, folder, *extra, env):
    return subprocess.run(
        [repo_python, str(REPO / "summarize.py"), str(folder), *extra],
        capture_output=True, env=env)


def norm(b: bytes) -> str:
    return b.decode("utf-8").replace("\r\n", "\n").rstrip()


def test_summarize_table_matches_golden(fixtures_dir, golden_dir, repo_python, utf8_env):
    """Таблица сводки на фикстурах совпадает с золотым файлом."""
    r = run_summarize(repo_python, fixtures_dir, env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    expected = (golden_dir / "summarize_fixtures.txt").read_text(encoding="utf-8")
    assert norm(r.stdout) == expected.rstrip()


def test_summarize_by_gesture_matches_golden(fixtures_dir, golden_dir, repo_python, utf8_env):
    """Разрез сводки по жестам совпадает с золотым файлом."""
    r = run_summarize(repo_python, fixtures_dir, "--by-gesture", env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    expected = (golden_dir / "summarize_fixtures_by_gesture.txt").read_text(encoding="utf-8")
    assert norm(r.stdout) == expected.rstrip()


def test_summarize_csv_verdict_column(tmp_path, fixtures_dir, repo_python, utf8_env):
    """CSV-отчёт содержит колонку pass_90 с булевыми вердиктами."""
    out = tmp_path / "report.csv"
    r = run_summarize(repo_python, fixtures_dir, "--csv", str(out), env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {row["pass_90"] for row in rows} == {"True", "False"}
    assert {row["position"] for row in rows} == {"грудь", "шнурок", "подставка"}


def test_summarize_zero_frames_group_no_crash(tmp_path, fixtures_dir, repo_python, utf8_env):
    """B1 закрыт: папка только из оборванных сессий (0 кадров) не роняет сводку."""
    shutil.copy(fixtures_dir / "fix_zero_frames.json", tmp_path)
    r = run_summarize(repo_python, tmp_path, env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "нет кадров" in r.stdout.decode("utf-8")


def test_summarize_skips_noise_by_default(fixtures_dir, repo_python, utf8_env):
    """Сессия «не жест» не входит в сводку этапа 0 и видна в stderr как пропуск."""
    r = run_summarize(repo_python, fixtures_dir, env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8")
    assert "fix_noise.json" in err and "не жест" in err
    assert "телефон" not in r.stdout.decode("utf-8")


def test_summarize_include_noise_adds_group(fixtures_dir, repo_python, utf8_env):
    """С флагом --include-noise фон этапа 1 попадает в разрез по жестам."""
    r = run_summarize(repo_python, fixtures_dir, "--by-gesture", "--include-noise", env=utf8_env)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "телефон" in r.stdout.decode("utf-8")


def test_summarize_empty_folder_fails(tmp_path, repo_python, utf8_env):
    """Пустая папка даёт ненужевой код выхода и понятное сообщение."""
    r = run_summarize(repo_python, tmp_path, env=utf8_env)
    assert r.returncode != 0
    assert "нет JSON" in r.stderr.decode("utf-8", "replace")
