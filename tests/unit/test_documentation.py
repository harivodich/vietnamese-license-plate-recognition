"""Kiểm thử quy ước mọi class và hàm đều phải có docstring giải thích."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
PYTHON_ROOTS = ("app", "scripts", "src", "tests")
DOCUMENTED_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def test_all_classes_and_functions_have_docstrings() -> None:
    """Quét Python AST và liệt kê chính xác vị trí class/hàm còn thiếu docstring."""
    missing: list[str] = []
    for root_name in PYTHON_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, DOCUMENTED_NODES) and ast.get_docstring(node) is None:
                    relative_path = path.relative_to(PROJECT_ROOT)
                    missing.append(f"{relative_path}:{node.lineno}:{node.name}")

    assert not missing, "Thiếu docstring:\n" + "\n".join(sorted(missing))
