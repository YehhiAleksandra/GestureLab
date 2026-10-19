import json
import subprocess
from pathlib import Path

import numpy as np

import export_dataset
import train

REPO = Path(__file__).resolve().parent.parent


def run(script, repo_python, env, *argv):
    return subprocess.run([repo_python, str(REPO / script), *map(str, argv)],
                          capture_output=True, env=env)


def export(repo_python, env, folder, out, *extra):
    return run("export_dataset.py", repo_python, env, folder, "--out", out, *extra)


def test_export_splits_by_person(fixtures_dir, tmp_path, repo_python, utf8_env):
    """Датасет делится train/val по исполнителям: один человек не попадает в оба сета."""
    out = tmp_path / "ds.npz"
    r = export(repo_python, utf8_env, fixtures_dir, out)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    d = np.load(out, allow_pickle=True)
    tr, va = set(d["person"][~d["is_val"]]), set(d["person"][d["is_val"]])
    assert tr and va and not (tr & va)
    t = int(round(1.5 * 15))
    assert d["X"].shape[1:] == (t, export_dataset.FEATURE_DIM)
    assert "телефон" not in [str(n) for n in d["names"]]


def test_export_prints_the_file_it_wrote(fixtures_dir, tmp_path, repo_python, utf8_env):
    """Без .npz в --out export печатает реальное имя файла: np.savez дописывает расширение (B24)."""
    r = export(repo_python, utf8_env, fixtures_dir, tmp_path / "ds")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    printed = [line.split("датасет: ", 1)[1].strip()
               for line in r.stdout.decode("utf-8").splitlines() if "датасет:" in line]
    assert len(printed) == 1, r.stdout.decode("utf-8")
    path = Path(printed[0])
    assert path.exists(), f"напечатан путь, которого нет: {path}"
    np.load(path, allow_pickle=True)


def test_export_single_person_splits_by_session(fixtures_dir, tmp_path, repo_python, utf8_env):
    """При одном исполнителе сплит идёт по сессиям и печатает предупреждение об утечке."""
    folder = tmp_path / "one"
    folder.mkdir()
    for name in ("fix_both_hands.json", "fix_both_hands_shnurok.json"):
        data = json.loads((fixtures_dir / name).read_text(encoding="utf-8"))
        data["person"] = "В"
        (folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "one.npz"
    r = export(repo_python, utf8_env, folder, out)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert "исполнитель один" in r.stdout.decode("utf-8")
    d = np.load(out, allow_pickle=True)
    assert d["is_val"].any() and (~d["is_val"]).any()


def synthetic_dataset(n_per_class=24, t=12, seed=7):
    """Два класса: движется разная координата кисти, плюс шум и смещение запястья."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in (0, 1):
        for _ in range(n_per_class):
            w = np.zeros((t, 126), np.float32)
            w[:, 0], w[:, 63] = 0.3, 0.7
            w[:, 3 + c] = np.linspace(0.0, 0.4, t)
            w[:, 66 + c] = np.linspace(0.0, -0.4, t)
            X.append(w + rng.normal(0, 0.01, w.shape).astype(np.float32))
            y.append(c)
    X = np.array(X, np.float32)
    y = np.array(y, np.int64)
    is_val = np.zeros(len(y), bool)
    is_val[::4] = True
    return X, y, is_val


def test_train_learns_separable_synthetic_classes():
    """GRU-baseline разделяет два синтетических класса: top-1 не ниже 80% на train и val."""
    res = train.train_model(*synthetic_dataset(), epochs=30, hidden=8, seed=3)
    assert res["train_top1"] >= 80.0
    assert res["val_top1"] >= 80.0
    assert res["val_top3"] >= res["val_top1"]


def test_export_then_train_pipeline(fixtures_dir, tmp_path, repo_python, utf8_env):
    """Связка export_dataset.py -> train.py проходит на фикстурах и печатает top-1/top-3."""
    ds = tmp_path / "ds.npz"
    assert export(repo_python, utf8_env, fixtures_dir, ds).returncode == 0
    model = tmp_path / "model.npz"
    r = run("train.py", repo_python, utf8_env, ds, "--epochs", 3, "--hidden", 4,
            "--save", model)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8")
    assert "train top-1" in out and "val   top-1" in out
    assert model.exists()
