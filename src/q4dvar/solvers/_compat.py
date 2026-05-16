from __future__ import annotations

from importlib import import_module
from types import ModuleType


def export_public(module_name: str, target_globals: dict[str, object]) -> None:
    """Re-export public names from a module whose path is not import-syntax friendly."""

    module: ModuleType = import_module(module_name)
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in vars(module) if not name.startswith("_")]

    for name in public_names:
        target_globals[name] = getattr(module, name)
    target_globals["__all__"] = list(public_names)
