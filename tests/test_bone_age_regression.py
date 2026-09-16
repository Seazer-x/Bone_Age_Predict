from bone_age.bone_age import calc_bone_age


def test_calc_bone_age_returns_reasonable_value():
    for sex in ("boy", "girl"):
        age = calc_bone_age(300, sex)
        assert isinstance(age, float)
        assert 0 < age < 30


def test_invalid_sex_rejected():
    try:
        calc_bone_age(300, "unknown")
    except ValueError:
        return
    raise AssertionError("invalid sex should raise ValueError")
