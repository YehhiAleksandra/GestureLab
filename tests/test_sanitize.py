from gesturelab import safe_name


def test_safe_name_strips_path_and_spaces():
    """Слэши, пробелы и пунктуация становятся подчёркиваниями."""
    assert safe_name("да / нет! (тест)") == "да_нет_тест"


def test_safe_name_empty_falls_back():
    """Пустая метка жеста не даёт пустое имя файла."""
    assert safe_name("") == "жест"
    assert safe_name("   ") == "жест"


def test_safe_name_keeps_cyrillic_and_underscore():
    """Кириллица и подчёркивания словаря v1 сохраняются."""
    assert safe_name("повторите_пожалуйста") == "повторите_пожалуйста"
