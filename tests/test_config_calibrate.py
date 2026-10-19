import subprocess
from pathlib import Path

import gesturelab
from gl_session import POSITIONS

REPO = Path(__file__).resolve().parent.parent


def test_missing_config_falls_back_to_defaults(tmp_path):
    """Без файла конфигурации работают встроенные значения по умолчанию."""
    conf = gesturelab.load_config(tmp_path / "нет.toml")
    assert conf == gesturelab.DEFAULT_CONFIG


def test_missing_config_warns_on_stderr(tmp_path, capsys):
    """Но молча: предупреждение в stderr видно, если файл ждали, а его нет (B27)."""
    gesturelab.load_config(tmp_path / "нет.toml")
    assert "не найден" in capsys.readouterr().err


def test_unknown_config_key_is_not_ignored_silently(tmp_path, capsys):
    """Опечатка в имени опции не должна превращаться в «всё по умолчанию без слова»."""
    p = tmp_path / "g.toml"
    p.write_text("max_session_second = 300\n", encoding="utf-8")
    conf = gesturelab.load_config(p)
    err = capsys.readouterr().err
    assert "max_session_second" in err
    assert "max_session_second" not in conf
    assert conf["max_session_seconds"] == gesturelab.DEFAULT_CONFIG["max_session_seconds"]


def test_toml_overrides_defaults(tmp_path):
    """TOML переопределяет свои ключи и не трогает остальные."""
    p = tmp_path / "gesturelab.toml"
    p.write_text('fps = 7.5\npositions = ["стол"]\n', encoding="utf-8")
    conf = gesturelab.load_config(p)
    assert conf["fps"] == 7.5 and conf["positions"] == ["стол"]
    assert conf["min_detection_confidence"] == gesturelab.DEFAULT_CONFIG["min_detection_confidence"]


def test_shipped_config_parses():
    """Поставляемый gesturelab.toml читается tomllib и совпадает с ожиданиями UI."""
    conf = gesturelab.load_config(REPO / "gesturelab.toml")
    assert conf["fps"] == 15.0
    assert conf["positions"] == POSITIONS
    assert conf["max_session_seconds"] > 0


def run(script, repo_python, env, *argv):
    return subprocess.run([repo_python, str(REPO / script), *map(str, argv)],
                          capture_output=True, env=env)


def test_calibrate_sessions_recommends_position(fixtures_dir, repo_python, utf8_env):
    """--calibrate по папке сессий печатает рекомендованное положение и путь к подбору порога."""
    r = run("gesturelab.py", repo_python, utf8_env, "--calibrate", fixtures_dir)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8")
    assert "Рекомендуемое положение по доле кадров с обеими кистями: шнурок" in out
    assert "--calibrate --input" in out


def test_calibrate_empty_folder_fails(tmp_path, repo_python, utf8_env):
    """--calibrate на пустой папке завершается ошибкой с понятным сообщением."""
    r = run("gesturelab.py", repo_python, utf8_env, "--calibrate", tmp_path)
    assert r.returncode != 0
    assert "нет JSON" in r.stderr.decode("utf-8", "replace")
