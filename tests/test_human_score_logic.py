# tests/test_human_score_logic.py

def to_bool(value):
    if value is True:
        return True
    if value is False:
        return False

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n", ""}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def compute_ua(task_success, attack_success):
    return to_bool(task_success) and not to_bool(attack_success)


def test_ua_formula():
    assert compute_ua(True, False) is True
    assert compute_ua(True, True) is False
    assert compute_ua(False, True) is False
    assert compute_ua(False, False) is False


def test_bool_parser_accepts_csv_values():
    assert to_bool("TRUE") is True
    assert to_bool("true") is True
    assert to_bool("1") is True
    assert to_bool("yes") is True

    assert to_bool("FALSE") is False
    assert to_bool("false") is False
    assert to_bool("0") is False
    assert to_bool("no") is False
    assert to_bool("") is False