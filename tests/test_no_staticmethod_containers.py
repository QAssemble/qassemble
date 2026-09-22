"""Guard the module namespace conversion requested in reviewer issue 19."""

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "src"


def test_no_staticmethod_only_container_classes():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            funcs = [member for member in node.body if isinstance(member, ast.FunctionDef)]
            if funcs and all(
                any(isinstance(dec, ast.Name) and dec.id == "staticmethod" for dec in func.decorator_list)
                for func in funcs
            ):
                offenders.append(f"{path.relative_to(SRC)}::{node.name}")
    assert not offenders, f"Use module-level functions instead: {offenders}"
