"""Static dependency contracts for the inference package layout."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def package_imports(package):
    imports = set()
    for path in (ROOT / "rvc" / package).rglob("*.py"):
        imports.update(imported_modules(path))
    return imports


def test_core_does_not_depend_on_inference_domains():
    imports = package_imports("core")
    assert not any(name == "rvc.vc" or name.startswith("rvc.vc.") for name in imports)
    assert not any(name == "rvc.separation" or name.startswith("rvc.separation.") for name in imports)


def test_inference_domains_do_not_depend_on_each_other():
    vc_imports = package_imports("vc")
    separation_imports = package_imports("separation")
    assert not any(name == "rvc.separation" or name.startswith("rvc.separation.") for name in vc_imports)
    assert not any(name == "rvc.vc" or name.startswith("rvc.vc.") for name in separation_imports)
