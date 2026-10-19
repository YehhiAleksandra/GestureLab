import gl_session
import session_schema


def test_valid_fixtures_pass_schema(fixtures_dir):
    """Валидные фикстуры проходят проверку схемы без ошибок."""
    for name in ("fix_both_hands.json", "fix_one_hand.json", "fix_face_only.json",
                 "fix_both_hands_shnurok.json"):
        data, errs = session_schema.read_session(fixtures_dir / name)
        assert errs == [], f"{name}: {errs}"
        assert data is not None


def test_missing_face_flag_is_schema_error(fixtures_dir):
    """Кадр без флага face — ошибка схемы, а не молчаливый ноль."""
    _, errs = session_schema.read_session(fixtures_dir / "fix_no_face_flag.json")
    assert any("face" in e for e in errs)


def test_broken_json_is_read_error(fixtures_dir):
    """Битый файл возвращается как ошибка чтения, а не как исключение."""
    data, errs = session_schema.read_session(fixtures_dir / "fix_broken.json")
    assert data is None and errs


def test_missing_version_is_error():
    """Сессия без поля version отклоняется схемой."""
    errs = session_schema.validate_session(
        {"gesture": "да", "position": "грудь", "person": "В", "frames": [], "summary": {}})
    assert any("version" in e for e in errs)


def test_landmark_vector_must_be_63():
    """Вектор точек не из 63 чисел отклоняется схемой."""
    bad = {"version": 1, "gesture": "да", "position": "грудь", "person": "В",
           "frames": [{"t_ms": 0, "hands": 1, "face": True,
                       "hand_landmarks": [[0.1] * 62]}],
           "summary": {"frames": 1, "both_hands_pct": 0.0, "face_pct": 100.0}}
    assert any("63" in e for e in session_schema.validate_session(bad))


def test_v2_requires_kind():
    """В схеме v2 поле kind обязательно: без него жест не отличить от фона."""
    bad = {"version": 2, "gesture": "да", "position": "грудь", "person": "В",
           "frames": [], "summary": {"frames": 0, "both_hands_pct": 0.0, "face_pct": 0.0}}
    assert any("kind" in e for e in session_schema.validate_session(bad))


def test_v1_session_migrates_to_gesture(fixtures_dir):
    """Сессия v1 без kind читается как жест, recorded_frames — как длина frames."""
    data, errs = session_schema.read_session(fixtures_dir / "fix_both_hands.json")
    assert errs == []
    assert gl_session.kind_of(data) == gl_session.KIND_GESTURE
    assert gl_session.recorded_frames_of(data) == len(data["frames"])


def test_noise_session_filename_is_marked():
    """Сессия «не жест» помечается в имени файла и не смешивается с жестами."""
    noise = gl_session.session_filename(
        {"gesture": "телефон", "position": "грудь", "kind": gl_session.KIND_NOISE})
    assert "_ne_zhest_телефон_грудь.json" in noise
    gest = gl_session.session_filename(
        {"gesture": "да", "position": "грудь", "kind": gl_session.KIND_GESTURE})
    assert "ne_zhest" not in gest and gest.endswith("_да_грудь.json")
