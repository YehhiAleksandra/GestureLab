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
