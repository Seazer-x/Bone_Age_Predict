import ast
import math
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "bone_age" / "bone_age.py"


def load_calc_bone_age():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calc_bone_age"
    )
    namespace = {"math": math}
    module = ast.Module(body=[func], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["calc_bone_age"]


def test_calc_bone_age_returns_reasonable_value():
    calc_bone_age = load_calc_bone_age()
    for sex in ("boy", "girl"):
        age = calc_bone_age(300, sex)
        assert isinstance(age, float)
        assert 0 < age < 30


def test_invalid_sex_rejected():
    calc_bone_age = load_calc_bone_age()
    try:
        calc_bone_age(300, "unknown")
    except ValueError:
        return
    raise AssertionError("invalid sex should raise ValueError")
